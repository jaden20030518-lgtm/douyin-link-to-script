"""Validate the report's anonymous aggregate arithmetic. Python standard library only."""
import json
from pathlib import Path

root=Path(__file__).resolve().parent
strategies=json.loads((root/'data/metrics.json').read_text(encoding='utf-8'))['strategies']
rows=json.loads((root/'data/per-video.json').read_text(encoding='utf-8'))['rows']
assert len(strategies)==7 and len(rows)==70
for strategy in strategies:
    group=[r for r in rows if r['arm']==strategy['id']]
    assert len(group)==10 and len({r['case'] for r in group})==10
    assert sum(r['anchors'] for r in group)==strategy['subtitle_checkpoint_count']==72
    assert sum(r['chars'] for r in group)==strategy['normalized_reference_characters']==613
    assert sum(r['number_occurrences'] for r in group)==strategy['numeric_checkpoint_occurrences']==25
    assert sum(r['exact'] for r in group)==strategy['exact_checkpoint_matches']
    assert sum(r['edits'] for r in group)==strategy['character_edits']
    assert sum(r['number_present'] for r in group)==strategy['unambiguous_numeric_matches']
    expected=100*sum(r['exact'] for r in group)/72
    assert abs(strategy['exact_checkpoint_match_pct']-expected)<0.0001
    expected=100*(1-sum(r['edits'] for r in group)/613)
    assert abs(strategy['checkpoint_character_agreement_pct']-expected)<0.0001
    print(strategy['id'],strategy['exact_checkpoint_matches'],'/72',strategy['unambiguous_numeric_matches'],'/25')
print('Validated: 7 strategies, 10 videos each, 72 checkpoints, 613 characters, 25 numeric occurrences.')
