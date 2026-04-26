class CurriculumScheduler:
    """Start on easy, promote when agent achieves >=0.75 score 3 runs in a row.

    Bug E fix: LEVELS now uses the EXACT keys registered in
    `incident_env.server.scenarios.SCENARIOS`. Previously this class returned
    bare names like "db_failover" which don't exist in SCENARIOS (it's
    registered as "hard_db_failover"), causing reset() to raise ValueError.
    """

    LEVELS = [
        "easy",
        "medium",
        "hard",
        "easy_dns_propagation",
        "easy_redis_oom",
        "medium_cert_expiry",
        "medium_k8s_eviction",
        "hard_regex_catastrophe",
        "hard_db_failover",
        "hard_s3_keyspace_overflow",
    ]

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
