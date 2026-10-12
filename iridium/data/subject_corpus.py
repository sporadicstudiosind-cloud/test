"""Real labelled subject data. Network access happens only when called."""
import hashlib

from ..codecs.spans import Sample, text_span
from ..runtime.chat import BOS, EOS, ROLE_TOKEN, _control
from ..training.datasets import Corpus
from ..training.tasks import Item


RECIPES = {
    'mathematics_symbolic_proof': ('openai/gsm8k', 'main', 'mit'),
    'code_systems_tools': ('google-research-datasets/mbpp', 'full', 'cc-by-4.0'),
    'science_physics_simulation': ('allenai/ai2_arc', 'ARC-Challenge', 'cc-by-sa-4.0'),
}


def real_subject_corpus(subjects, per_subject=256, max_tokens=1024):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    if per_subject < 1:
        raise ValueError('per_subject must be positive')
    items, provenance = [], []
    for subject in subjects:
        if subject not in RECIPES:
            continue
        name, subset, license_name = RECIPES[subject]
        revision = HfApi().dataset_info(name).sha
        data = load_dataset(name, subset, split='train', streaming=True, revision=revision)
        accepted = 0
        for index, row in enumerate(data):
            if subject == 'mathematics_symbolic_proof':
                question, answer = row['question'], row['answer']
            elif subject == 'code_systems_tools':
                question, answer = row['text'], row['code']
            else:
                choices = dict(zip(row['choices']['label'], row['choices']['text']))
                question = row['question'] + '\n' + '\n'.join(f'{k}: {v}' for k, v in choices.items())
                answer = row['answerKey'] + ': ' + choices[row['answerKey']]
            spans = [_control(BOS), _control(ROLE_TOKEN['user']), text_span(question, False, 16),
                     _control(ROLE_TOKEN['assistant']), text_span(answer, True, 16), _control(EOS, True)]
            identity = hashlib.sha256((name + '\0' + question + '\0' + answer).encode()).hexdigest()
            sample = Sample(spans, meta={'subject': subject, 'family': subject,
                'source': name, 'license': license_name, 'revision': revision, 'id': identity})
            if len(sample) <= max_tokens:
                items.append(Item(sample, subject))
                accepted += 1
            if accepted >= per_subject:
                break
        if not accepted:
            raise ValueError(f'{name}: no records fit max_tokens={max_tokens}')
        provenance.append({'dataset': name, 'revision': revision, 'split': 'train',
                           'subject': subject, 'license': license_name, 'accepted': accepted})
    if not items:
        raise ValueError('no configured subject recipes')
    return Corpus(items, 'train'), provenance
