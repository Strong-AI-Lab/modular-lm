
from .cluster import TokenLevelCluster, DiffTokenLevelCluster, InputLevelCluster, DiffInputLevelCluster
from .quantizer import TokenQuantizer, InputQuantizer



ROUTERS = {
    "TokenLevelCluster": TokenLevelCluster,
    "DiffTokenLevelCluster": DiffTokenLevelCluster,
    "InputLevelCluster": InputLevelCluster,
    "DiffInputLevelCluster": DiffInputLevelCluster,
    "TokenQuantizer": TokenQuantizer,
    "InputQuantizer": InputQuantizer,
}

def load_router(router_name : str, router_config : dict, router_path : str = None):
    if router_name not in ROUTERS:
        raise ValueError(f"Unknown router name: {router_name}")
    
    routing_strategy = ROUTERS[router_name](**router_config)
    if router_path is not None:
        routing_strategy.load_strategy(router_path)

    return routing_strategy


    