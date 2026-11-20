"""Use a TRUSTED trained Iridium checkpoint; no pretrained external model needed.

python examples/media_workflow.py --checkpoint runs/studio/studio-100m-final.pt \
    --source source.mp4 --start 10 --end 20 --output media_outputs
"""
import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iridium.training.trainer import load_checkpoint
from iridium.agency.media_agent import MediaTools, MediaAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--start', type=float, default=0)
    parser.add_argument('--end', type=float, required=True)
    parser.add_argument('--output', default='media_outputs')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--request', help='Use learned JSON tool planning instead of the fixed workflow')
    args = parser.parse_args()
    model, manifest = load_checkpoint(args.checkpoint, args.device)
    if (manifest.get('data', {}).get('media_format') != 'iridium-rgb-pcm-v1'
            or not (manifest.get('data', {}).get('media_examples') or manifest.get('data', {}).get('families', {}).get('multimodal'))):
        raise RuntimeError('Checkpoint does not declare the RGB/PCM training format. Train paired media first.')
    if args.request:
        result = MediaAgent(model, args.output).run(args.request, [args.source])
    else:
        result = MediaTools(model, args.output).clip_with_subtitles(args.source, args.start, args.end)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
