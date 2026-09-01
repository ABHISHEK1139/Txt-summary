import os
from src.summarizer import summarize_with_local_model, _split_into_sections

fixtures_dir = os.path.join(os.path.dirname(__file__), 'fixtures')
text = open(os.path.join(fixtures_dir, 'test_academic.txt')).read()
sections = _split_into_sections(text * 2)
print('CHUNKS DETECTED:')
for i, s in enumerate(sections):
    print(f'-- Chunk {i} --')
    print(s[:150] + '...')
