from createDB import get_all
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def get_relevant_chunks(query_embedding, top_k=5):

    results = []

    rows = get_all()
    
    print(len(rows))
    done=1
    for row in rows:
        print(done)

        content = row[0]

        embedding = np.frombuffer(
            row[1],
            dtype=np.float32
        ).reshape(1, -1)

        similarity = cosine_similarity(
            query_embedding,
            embedding
        )[0][0]

        results.append((
            content,
            embedding,
            similarity
        ))
        done+=1

    results.sort(
        key=lambda x: x[2],
        reverse=True
    )

    return results[:top_k]