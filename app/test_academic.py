from summarizer import summarize_with_local_model, _split_into_sections
text = open('test_academic.txt').read()
sections = _split_into_sections(text*2)
print('CHUNKS DETECTED:')
for i, s in enumerate(sections):
    print(f'-- Chunk {i} --')
    print(s[:150] + '...')
