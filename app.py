

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

sentences = [
    'The new movie is awesome',
    'This recent movie is so good',
]

model = SentenceTransformer('Supabase/gte-small')
embeddings = model.encode(sentences)
print(cos_sim(embeddings[0], embeddings[1]))
