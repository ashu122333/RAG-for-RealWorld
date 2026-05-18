from singleHopChunkFinder import get_relevant_chunks
from generateAnswer import generate_response
from inputValidation import isValidInput
from inputRewritting import rewrite_query
from outputValidation import isValidOutput
from checkSingleOrMulti import generate_response as check_single_multi
from multiHopChunkFinder import relevante_chunks
import json



def main(query):
    # validate input query
    is_valid = json.loads(isValidInput(query))
    if not is_valid["is_valid"]:
        print(f"Query is not valid: {is_valid['reason']}")
        return "Invalid query. Please rephrase and try again."
    # query re-writing
    query = json.loads(rewrite_query(query))["new_query"]

    # single or multi-hop retrival
    hop_type=json.loads(check_single_multi(query))
    if hop_type["type"]=="multi_hop":
        print(hop_type["sub_questions"])
        chunks=relevante_chunks(hop_type["sub_questions"])
        print(len(chunks))
        
    else:
        chunks = get_relevant_chunks(query)
        print(query)    
        print(len(chunks))    
    # generating context    
    context = "\n".join([f"Summary: {c['summary']}\nChunk: {c['text']}" for c in chunks])

    #generate answer
    answer = generate_response(query, context)
    print("Generated Answer:", json.loads(answer)["answer"])

    #valide output
    is_valid_output = json.loads(isValidOutput(query, context, answer))
    if not is_valid_output["grounded"]:
        print(f"Output is not valid: {is_valid_output['grounded_unsupported_claims']}")
        # return "Could not generate a valid answer based on the retrieved information."
    if not is_valid_output["relevant"]:
        print(f"Output is not relevant: {is_valid_output['relevant_explanation']}")
        # return "Could not generate a relevant answer based on the retrieved information."

    return json.loads(answer)["answer"]


# if __name__ == "__main__":
#     query="What modules make up the RAG workflow studied in 'Searching for Best Practices in RAG', and what are some recommended choices?"
#     print(main(query))
