#!/usr/bin/env node
import { spawn, spawnSync } from "node:child_process";
import { mkdirSync } from "node:fs";
import { join } from "node:path";
import { homedir, tmpdir } from "node:os";
import process from "node:process";

const DEFAULT_PORT = 9333;
const DEFAULT_PROFILE = join(tmpdir(), "douyin_chrome_profile");
const DEFAULT_OUTDIR = join(homedir(), "Downloads");
const USER_AGENT =
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36";

function usage(exitCode = 2) {
  console.error(`Usage: douyin_download.mjs <douyin-url> [--output-dir DIR] [--profile-dir DIR] [--port N] [--no-open] [--timeout MS]`);
  process.exit(exitCode);
}

function parseArgs(argv) {
  const opts = {
    outputDir: DEFAULT_OUTDIR,
    profileDir: DEFAULT_PROFILE,
    port: DEFAULT_PORT,
    openChrome: true,
    timeoutMs: 20000,
  };
  let url = null;
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--output-dir") opts.outputDir = argv[++i] || usage();
    else if (arg === "--profile-dir") opts.profileDir = argv[++i] || usage();
    else if (arg === "--port") opts.port = Number(argv[++i] || usage());
    else if (arg === "--timeout") opts.timeoutMs = Number(argv[++i] || usage());
    else if (arg === "--no-open") opts.openChrome = false;
    else if (arg === "--help" || arg === "-h") usage(0);
    else if (!url) url = arg;
    else usage();
  }
  if (!url) usage();
  return { url, opts };
}

async function sleep(ms) {
  await new Promise((resolve) => setTimeout(resolve, ms));
}

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} ${url}`);
  return res.json();
}

async function cdpAvailable(port) {
  try {
    await fetchJson(`http://127.0.0.1:${port}/json/version`);
    return true;
  } catch {
    return false;
  }
}

function openChrome(url, opts) {
  mkdirSync(opts.profileDir, { recursive: true });
  const args = [
    "-na",
    "Google Chrome",
    "--args",
    `--user-data-dir=${opts.profileDir}`,
    `--remote-debugging-port=${opts.port}`,
    "--no-first-run",
    "--no-default-browser-check",
    url,
  ];
  const child = spawn("open", args, { stdio: "ignore", detached: true });
  child.unref();
}

async function ensureChrome(url, opts) {
  if (await cdpAvailable(opts.port)) return;
  if (!opts.openChrome) throw new Error(`Chrome CDP is not available on port ${opts.port}`);
  openChrome(url, opts);
  for (let i = 0; i < 40; i++) {
    await sleep(250);
    if (await cdpAvailable(opts.port)) return;
  }
  throw new Error(`Timed out waiting for Chrome CDP on port ${opts.port}`);
}

async function newPage(port, url) {
  const res = await fetch(`http://127.0.0.1:${port}/json/new?${encodeURIComponent(url)}`, { method: "PUT" });
  if (!res.ok) throw new Error(`Failed to open Chrome tab: HTTP ${res.status}`);
  return res.json();
}

function sendFactory(ws) {
  let id = 0;
  const pending = new Map();
  ws.addEventListener("message", (ev) => {
    const msg = JSON.parse(ev.data);
    if (!msg.id || !pending.has(msg.id)) return;
    const { resolve, reject } = pending.get(msg.id);
    pending.delete(msg.id);
    if (msg.error) reject(new Error(JSON.stringify(msg.error)));
    else resolve(msg.result);
  });
  return (method, params = {}) =>
    new Promise((resolve, reject) => {
      const msgId = ++id;
      pending.set(msgId, { resolve, reject });
      ws.send(JSON.stringify({ id: msgId, method, params }));
    });
}

async function captureDetail(page, targetUrl, timeoutMs) {
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    ws.onopen = resolve;
    ws.onerror = reject;
  });

  const send = sendFactory(ws);
  let settled = false;
  let detailRequestId = null;

  async function readDetailBody(requestId, resolve) {
    if (settled) return;
    settled = true;
    try {
      const bodyResp = await send("Network.getResponseBody", { requestId });
      if (!bodyResp.body) throw new Error("Empty aweme detail response body");
      resolve(JSON.parse(bodyResp.body));
    } catch (error) {
      resolve({ error: String(error) });
    } finally {
      ws.close();
    }
  }

  const detailPromise = new Promise((resolve) => {
    ws.addEventListener("message", async (ev) => {
      const msg = JSON.parse(ev.data);
      if (settled) return;
      if (msg.method === "Network.responseReceived") {
        const responseUrl = msg.params?.response?.url || "";
        if (!responseUrl.includes("/aweme/v1/web/aweme/detail/")) return;
        detailRequestId = msg.params.requestId;
        setTimeout(() => readDetailBody(detailRequestId, resolve), 400);
      } else if (msg.method === "Network.loadingFinished" && msg.params?.requestId === detailRequestId) {
        await readDetailBody(detailRequestId, resolve);
      }
    });
  });

  await send("Network.enable");
  await send("Page.enable");
  await send("Page.navigate", { url: targetUrl });

  const timeout = new Promise((resolve) =>
    setTimeout(() => resolve({ error: "timeout", actionRequired: true }), timeoutMs),
  );
  const result = await Promise.race([detailPromise, timeout]);
  if (!settled) ws.close();
  return result;
}

function pickVideo(detail) {
  const aweme = detail?.aweme_detail;
  const video = aweme?.video || {};
  const chosen = video.play_addr_h264 || video.play_addr || video.download_addr || video.play_addr_265;
  const urls = chosen?.url_list || [];
  if (!aweme || !urls.length) return null;
  return {
    title: aweme.desc || aweme.item_title || aweme.aweme_id,
    awemeId: aweme.aweme_id,
    width: chosen.width,
    height: chosen.height,
    durationMs: aweme.duration || video.duration,
    size: chosen.data_size,
    url: urls[0],
  };
}

function sanitizeName(name) {
  return String(name)
    .replace(/[\\/:*?"<>|#%&{}$!'@+`=]/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 90) || "douyin-video";
}

function download(url, targetPath, referer) {
  const args = [
    "-L",
    "--fail",
    "--retry",
    "3",
    "-A",
    USER_AGENT,
    "-e",
    referer,
    "-o",
    targetPath,
    url,
  ];
  const result = spawnSync("curl", args, { stdio: "inherit" });
  if (result.status !== 0) throw new Error(`curl failed with exit code ${result.status}`);
}

function ffprobe(path) {
  const result = spawnSync(
    "ffprobe",
    [
      "-v",
      "error",
      "-show_entries",
      "format=duration,size:stream=codec_name,codec_type,width,height",
      "-of",
      "json",
      path,
    ],
    { encoding: "utf8" },
  );
  if (result.status !== 0) return null;
  try {
    return JSON.parse(result.stdout);
  } catch {
    return null;
  }
}

async function main() {
  const { url, opts } = parseArgs(process.argv.slice(2));
  mkdirSync(opts.outputDir, { recursive: true });

  await ensureChrome(url, opts);
  const page = await newPage(opts.port, url);
  const detail = await captureDetail(page, url, opts.timeoutMs);
  if (detail?.actionRequired) {
    console.error("ACTION_REQUIRED: Complete Douyin slider/login verification in the opened Chrome window, then rerun this command.");
    process.exit(3);
  }
  if (detail?.error) throw new Error(detail.error);

  const video = pickVideo(detail);
  if (!video) throw new Error("Could not find a playable video URL in aweme detail response");

  const filename = `${sanitizeName(video.title)}-${video.awemeId}.mp4`;
  const outputPath = join(opts.outputDir, filename);
  download(video.url, outputPath, `https://www.douyin.com/video/${video.awemeId}`);
  const probe = ffprobe(outputPath);
  console.log(
    JSON.stringify(
      {
        outputPath,
        title: video.title,
        awemeId: video.awemeId,
        expected: {
          width: video.width,
          height: video.height,
          durationMs: video.durationMs,
          size: video.size,
        },
        ffprobe: probe,
      },
      null,
      2,
    ),
  );
}

main().catch((error) => {
  console.error(`ERROR: ${error.message}`);
  process.exit(1);
});
