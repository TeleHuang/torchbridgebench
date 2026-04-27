import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

@dataclass
class TestCaseResult:
    test_name: str
    suite_name: str
    layer: str  # operator, module, model
    compatibility: bool
    correctness: Optional[bool] = None
    performance_ms: Optional[float] = None
    error_message: Optional[str] = None
    usability_score: Optional[int] = None
    skipped: bool = False

@dataclass
class BenchmarkReport:
    backend: str
    timestamp: str
    environment: Dict[str, Any]
    results: list[TestCaseResult]
    
    def to_json(self):
        return json.dumps(asdict(self), indent=2)

    def save(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
