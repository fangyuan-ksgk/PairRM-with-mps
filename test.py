import llm_blender_mps
import torch
device = torch.device("mps")

blender = llm_blender_mps.Blender()
blender.loadranker("llm-blender/PairRM", device=device) # load ranker checkpoint

inputs = ["hello, how are you!", "I love you!"]
candidates_texts = [["get out!", "hi! I am fine, thanks!", "bye!"], 
                    ["I love you too!", "I hate you!", "Thanks! You're a good guy!"]]
ranks = blender.rank(inputs, candidates_texts, return_scores=False, batch_size=1)
print(ranks)