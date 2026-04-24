# Scenarios package — pre-built failure scenarios
from incident_env.server.scenarios.easy import EasyScenario
from incident_env.server.scenarios.medium import MediumScenario
from incident_env.server.scenarios.hard import HardScenario
from incident_env.server.scenarios.dns_propagation import DnsPropagationScenario
from incident_env.server.scenarios.redis_memory_leak import RedisMemoryLeakScenario
from incident_env.server.scenarios.cert_expiry import CertExpiryScenario
from incident_env.server.scenarios.k8s_eviction import K8sEvictionScenario
from incident_env.server.scenarios.regex_catastrophe import RegexCatastropheScenario
from incident_env.server.scenarios.s3_keyspace import S3KeyspaceScenario
from incident_env.server.scenarios.db_failover import DbFailoverScenario

SCENARIOS = {
    # Original hackathon scenarios
    "easy": EasyScenario,
    "medium": MediumScenario,
    "hard": HardScenario,
    
    # Real-world postmortem scenarios
    "easy_dns_propagation": DnsPropagationScenario,
    "easy_redis_oom": RedisMemoryLeakScenario,
    "medium_cert_expiry": CertExpiryScenario,
    "medium_k8s_eviction": K8sEvictionScenario,
    "hard_regex_catastrophe": RegexCatastropheScenario,
    "hard_s3_keyspace_overflow": S3KeyspaceScenario,
    "hard_db_failover": DbFailoverScenario,
}

__all__ = ["SCENARIOS"]
