"""PolyTune greedy generation with native decoder KV caches.

Same float32 encoder, positional offsets, logits, argmax, EOS, and token budget
as the frozen upstream implementation. No output suppression or heuristics.
"""
import types
import torch

def generate(self,mistake_inputs,score_inputs,max_length=1024,output_hidden_states=False,**kwargs):
    hidden=self.encoder(a1=score_inputs,a2=mistake_inputs)
    batch=mistake_inputs.shape[0]
    ids=torch.full((batch,1),self.config.decoder_start_token_id,dtype=torch.long,device=self.device)
    unfinished=torch.ones(batch,dtype=torch.long,device=self.device);past=None
    for _ in range(max_length):
        decoded=self.decoder(input_ids=ids if past is None else ids[:,-1:],encoder_hidden_states=hidden,
                             past_key_values=past,use_cache=True,return_dict=True)
        logits=self.lm_head(decoded[0]);token=logits[:,-1,:].argmax(-1,keepdim=True)
        token=token*unfinished[:,None]+self.config.pad_token_id*(1-unfinished[:,None])
        unfinished[token[:,0]==self.config.eos_token_id]=0
        ids=torch.cat([ids,token],-1);past=decoded.past_key_values
        if unfinished.max()==0:break
    return (ids,hidden) if output_hidden_states else ids

def install(model):
    model.generate=types.MethodType(generate,model)
    return model
