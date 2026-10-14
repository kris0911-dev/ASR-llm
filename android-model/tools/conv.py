import json,struct,sys,glob,numpy as np
from pathlib import Path
from safetensors import safe_open
out=sys.argv[1]; d=Path(sys.argv[2] if len(sys.argv)>2 else 'buzz'); mel=Path(sys.argv[3] if len(sys.argv)>3 else 'mel_filters.npz')
conv_map={'self_attn.k_proj':'attn.key','self_attn.q_proj':'attn.query','self_attn.v_proj':'attn.value','self_attn.out_proj':'attn.out','self_attn_layer_norm':'attn_ln','encoder_attn.q_proj':'cross_attn.query','encoder_attn.v_proj':'cross_attn.value','encoder_attn.out_proj':'cross_attn.out','encoder_attn_layer_norm':'cross_attn_ln','fc1':'mlp.0','fc2':'mlp.2','final_layer_norm':'mlp_ln','encoder.layer_norm.bias':'encoder.ln_post.bias','encoder.layer_norm.weight':'encoder.ln_post.weight','encoder.embed_positions.weight':'encoder.positional_embedding','decoder.layer_norm.bias':'decoder.ln.bias','decoder.layer_norm.weight':'decoder.ln.weight','decoder.embed_positions.weight':'decoder.positional_embedding','decoder.embed_tokens.weight':'decoder.token_embedding.weight'}
def b2u():
    bs=list(range(33,127))+list(range(161,173))+list(range(174,256));cs=bs[:];n=0
    for b in range(256):
        if b not in bs: bs.append(b);cs.append(256+n);n+=1
    return dict(zip(bs,[chr(c) for c in cs]))
h=json.load(open(d/'config.json'));tokens=json.load(open(d/'vocab.json',encoding='utf8'))
ml=h.get('max_length') or h.get('max_target_positions',448)
f=open(out,'wb');P=lambda *v:f.write(struct.pack('i'*len(v),*v))
P(0x67676d6c,h['vocab_size'],h['max_source_positions'],h['d_model'],h['encoder_attention_heads'],h['encoder_layers'],int(ml),h['d_model'],h['decoder_attention_heads'],h['decoder_layers'],h['num_mel_bins'],1)
fl=np.load(mel)[f"mel_{h['num_mel_bins']}"].astype(np.float32)
P(*fl.shape);fl.tofile(f)
bd={v:k for k,v in b2u().items()};P(len(tokens))
for k,_ in sorted(tokens.items(),key=lambda x:x[1]):
    t=bytes(bd[c] for c in k);P(len(t));f.write(t)
for fn in sorted(glob.glob(str(d/'*.safetensors'))):
  with safe_open(fn,'np') as s:
    for name in s.keys():
        if name=='proj_out.weight': continue
        nn=name.split('.')[1:]
        if nn[1]=='layers':
            nn[1]='blocks';k='.'.join(nn[3:-1])
            m=('attn.key' if nn[0]=='encoder' else 'cross_attn.key') if k=='encoder_attn.k_proj' else conv_map[k]
            nm='.'.join(nn[:3]+[m]+nn[-1:])
        else:
            nm='.'.join(nn);nm=conv_map.get(nm,nm)
        a=s.get_tensor(name).squeeze().astype(np.float16)
        if nm in('encoder.conv1.bias','encoder.conv2.bias'): a=a.reshape(a.shape[0],1)
        ft=1
        if a.ndim<2 or nm in('encoder.conv1.bias','encoder.conv2.bias','encoder.positional_embedding','decoder.positional_embedding'): a=a.astype(np.float32);ft=0
        e=nm.encode();P(a.ndim,len(e),ft);P(*a.shape[::-1]);f.write(e);a.tofile(f)
f.close();print('done')
