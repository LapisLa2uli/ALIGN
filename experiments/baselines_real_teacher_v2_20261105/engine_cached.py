"""Inference-only entrypoint for the token-parity-verified cache optimization."""
import engine
from cached_decode import install
original_build=engine.build

def build(flavor,path):
    model=original_build(flavor,path)
    return install(model) if flavor=='polytune' else model

if __name__=='__main__':
    engine.build=build
    engine.main()
