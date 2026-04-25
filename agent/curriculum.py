class CurriculumScheduler:
    """Start on easy, promote when agent achieves >0.7 score 3 runs in a row."""
    
    LEVELS = ["easy", "medium", "hard", 
              "db_failover", "cert_expiry", "redis_memory_leak",
              "k8s_eviction", "dns_propagation", "regex_catastrophe", "s3_keyspace"]
    
    def __init__(self):
        self.current_level = 0
        self.consecutive_wins = 0
    
    def next_task(self) -> str:
        return self.LEVELS[self.current_level]
    
    def record_score(self, score: float):
        if score >= 0.75:
            self.consecutive_wins += 1
            if self.consecutive_wins >= 3 and self.current_level < len(self.LEVELS) - 1:
                self.current_level += 1
                self.consecutive_wins = 0
        else:
            self.consecutive_wins = 0
