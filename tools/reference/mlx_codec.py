"""Offline FP32 MLX adaptation of pinned NCM vae/vqvae/quant inference semantics.

Sudoku-easy only; authenticated configuration/weights, no training or fallback.
Independent implementation for same-device measurement, not a replacement oracle.
"""
import importlib.metadata
import math
import mlx.core as mx
from mlx_codec_assets import load_bundle
from mlx_operators import execute


def prepare_device():
    for package,version in (("mlx","0.32.2"),("mlx-metal","0.32.2"),("numpy","2.5.4")):
        if importlib.metadata.version(package)!=version:raise ValueError(f"unpinned {package}")
    if not mx.metal.is_available():raise RuntimeError("real Metal required; no fallback")
    mx.set_default_device(mx.gpu)
    one=mx.array([1.],dtype=mx.float32);mx.eval(one+one)


def save(trace,name,value):
    if trace is not None:
        if name in trace:raise ValueError(f"duplicate observation: {name}")
        trace[name]=value
    return value


def observe(trace,name,x,function):
    save(trace,name+".input",x)
    return save(trace,name+".output",function(x))


class Codec:
    def __init__(self,config,weights):
        self.config=config;self.w=weights
        self.length=config["context_length"];self.width=config["embed_dim"]
        self.heads=config["n_layers"]
        self.levels=config["vector_quantizer_config"]["levels"]["manual_levels"]
        self.ratio=config["vector_quantizer_config"]["quant_resi"]["ratio"]

    @classmethod
    def load(cls,root):
        config,host=load_bundle(root)
        weights={name:mx.array(array) for name,array in host.items()}
        mx.eval(*weights.values())
        return cls(config,weights)

    def linear(self,name,x,trace=None,label=""):
        def forward(x):
            y=x@self.w[name+".weight"].T
            bias=self.w.get(name+".bias")
            return y if bias is None else y+bias
        return observe(trace,label+name,x,forward)

    def norm(self,name,x,trace=None,label="",rms=False):
        def forward(x):
            value=execute({"kind":"rms_norm" if rms else "layer_norm","epsilon":1e-6},{"x":x})*self.w[name+".weight"]
            bias=self.w.get(name+".bias")
            return value if bias is None else value+bias
        return observe(trace,label+name,x,forward)

    def conv(self,name,x,stride=1,padding=0,transpose=False):
        w=self.w[name+".weight"]
        if transpose:
            y=mx.conv_transpose1d(x,w.transpose(1,2,0),stride=stride,padding=padding)
        elif padding==0 and x.shape[1]==w.shape[2]:
            # Equivalent FP32 projection for the cancellation-prone full-span sum.
            # Native MLX convolution exceeded the unchanged CUDA oracle budget.
            y=(x.transpose(0,2,1).reshape(x.shape[0],-1)@w.reshape(w.shape[0],-1).T)[:,None,:]
        else:
            y=mx.conv1d(x,w.transpose(0,2,1),stride=stride,padding=padding)
        return y+self.w[name+".bias"]

    def up(self,x,level):
        if level==0:return self.conv("vq.learned_up_l1.0",x,stride=self.length,transpose=True)
        if level==len(self.levels)-1:return x
        return execute({"kind":"linear1d","length":self.length},{"x":x})

    def phi(self,x,level):
        return (1-self.ratio)*x+self.ratio*self.conv(f"vq.quant_resi.{level}",x,padding=1)

    def encode(self,ids,trace=None):
        if ids.ndim!=2 or ids.shape[1]!=self.length:raise ValueError("invalid input shape")
        latent=observe(trace,"encode.encoder",ids,lambda x:self.w["encoder.wte.weight"][x]+self.w["encoder.wpe.weight"][mx.arange(self.length)])
        residual=execute({"kind":"group_norm","groups":self.config["pre_quant_groupnorm"],"epsilon":1e-5},{"x":latent})
        save(trace,"encode.pre_quant",residual)
        total=mx.zeros_like(residual);snapshots=[];codes=[]
        for level,length in enumerate(self.levels):
            prefix=f"encode.level_{level:02}"
            save(trace,prefix+".residual_before",residual)
            if level==0:down=self.conv("vq.learned_down_l1.0",residual,stride=self.length)
            elif level==len(self.levels)-1:down=residual
            else:down=execute({"kind":"area1d","length":length},{"x":residual})
            flat=down.reshape(-1,self.width);save(trace,prefix+".quant_input",flat)
            book=self.w[f"vq.embeddings.{level}.weight"]
            distances=mx.sum(mx.square(flat),axis=1,keepdims=True)+mx.sum(mx.square(book),axis=1)-2*(flat@book.T)
            save(trace,prefix+".distances",distances)
            selected=mx.argmin(distances,axis=1).reshape(ids.shape[0],length)
            up=self.up(book[selected],level);save(trace,prefix+".upsampled",up)
            contribution=self.phi(up,level);save(trace,prefix+".contribution",contribution)
            total=total+contribution;save(trace,prefix+".cumulative",total)
            residual=residual-contribution;snapshots.append(total);codes.append(selected)
        codes=mx.concatenate(codes,axis=1);save(trace,"encode.codes",codes)
        # Unused upstream z_levels outputs are intentionally omitted from this adapted API.
        return codes,snapshots

    def block(self,x,trace=None,label=""):
        prefix="decoder.middle_block"
        save(trace,label+prefix+".input",x)
        norm=self.norm(prefix+".ln_1",x,trace,label)
        save(trace,label+prefix+".attn.input",norm)
        qkv=self.linear(prefix+".attn.c_attn",norm,trace,label)
        q,k,v=[part.reshape(x.shape[0],x.shape[1],self.heads,-1).transpose(0,2,1,3) for part in mx.split(qkv,3,axis=2)]
        q=self.norm(prefix+".attn.q_norm",q,trace,label,rms=True)
        k=self.norm(prefix+".attn.k_norm",k,trace,label,rms=True)
        scores=(q@k.swapaxes(-1,-2))/math.sqrt(q.shape[-1])
        attention=(mx.softmax(scores,axis=-1,precise=True)@v).transpose(0,2,1,3).reshape(x.shape)
        attention=self.linear(prefix+".attn.c_proj",attention,trace,label)
        save(trace,label+prefix+".attn.output",attention)
        residual=x+attention
        norm=self.norm(prefix+".ln_2",residual,trace,label)
        save(trace,label+prefix+".mlp.input",norm)
        hidden=self.linear(prefix+".mlp.c_fc",norm,trace,label)
        hidden=observe(trace,label+prefix+".mlp.gelu",hidden,lambda h:0.5*h*(1+mx.erf(h/math.sqrt(2))))
        mlp=self.linear(prefix+".mlp.c_proj",hidden,trace,label)
        save(trace,label+prefix+".mlp.output",mlp)
        return save(trace,label+prefix+".output",residual+mlp)

    def decode(self,ids,trace=None,label="decode"):
        if ids.ndim!=2 or ids.shape[1]!=sum(self.levels):raise ValueError("invalid code shape")
        total=mx.zeros((ids.shape[0],self.length,self.width),dtype=mx.float32)
        offset=0;snapshots=[]
        for level,length in enumerate(self.levels):
            prefix=f"{label}.level_{level:02}"
            q=self.w[f"vq.embeddings.{level}.weight"][ids[:,offset:offset+length]];offset+=length
            up=self.up(q,level);save(trace,prefix+".upsampled",up)
            contribution=self.phi(up,level);save(trace,prefix+".contribution",contribution)
            total=total+contribution;save(trace,prefix+".cumulative",total);snapshots.append(total)
        save(trace,label+".decoder.input",total)
        hidden=self.block(total,trace,label+".")
        hidden=self.norm("decoder.ln_f",hidden,trace,label+".")
        logits=self.linear("decoder.output_layer",hidden,trace,label+".")
        save(trace,label+".decoder.output",logits);save(trace,label+".logits",logits)
        output=mx.argmax(logits,axis=-1);save(trace,label+".ids",output)
        return logits,output,snapshots
