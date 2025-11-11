import threading
from dataclasses import dataclass , field
from typing import Dict , List , Optional


@dataclass
class MinerState :
    shutdown_flag: bool = False
    thread_running_flags: List[bool] = field(default_factory = lambda : [False , False])
    local_height: int = 0
    height_to_best_difficulty: Dict[int , float] = field(default_factory = dict)
    updated_prev_hash: Optional[str] = None
    job_id: Optional[str] = None
    prev_hash: Optional[str] = None
    coinbase_part1: Optional[str] = None
    coinbase_part2: Optional[str] = None
    merkle_branch: Optional[List[str]] = None
    version: Optional[str] = None
    nbits: Optional[str] = None
    ntime: Optional[str] = None
    clean_jobs: Optional[bool] = None
    subscription_details: Optional[str] = None
    extranonce1: Optional[str] = None
    extranonce2_size: Optional[int] = None
    extranonce2: Optional[str] = None
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    
    def __post_init__(self):
        """Initialize lock after dataclass initialization"""
        if not hasattr(self, '_lock'):
            self._lock = threading.Lock()


