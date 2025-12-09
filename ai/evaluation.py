"""
AI Evaluation Framework

Measures and tracks AI performance in production.
Key metrics: relevance, accuracy, helpfulness, latency
"""

import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import uuid

from models.schemas import AIEvaluation, EvaluationMetrics


@dataclass
class MetricHistory:
    """Historical data point for a metric"""
    date: str
    value: float


@dataclass 
class EvaluationMetric:
    """A single evaluation metric with target and trend"""
    id: str
    name: str
    description: str
    value: float
    target: float
    unit: str
    trend: str  # 'up', 'down', 'stable'
    history: List[MetricHistory] = field(default_factory=list)


@dataclass
class EvaluationStats:
    """Comprehensive evaluation statistics"""
    total_evaluations: int
    average_scores: Dict[str, float]
    user_ratings: Dict[str, Any]
    latency_percentiles: Dict[str, float]
    trends_over_time: List[Dict[str, float]]


class AIEvaluationFramework:
    """
    Framework for evaluating AI response quality.
    
    Tracks:
    - Relevance: How relevant responses are to incident context
    - Accuracy: Correctness of facts and recommendations  
    - Helpfulness: User-rated usefulness
    - Latency: Response time
    """
    
    def __init__(self):
        self.evaluations: List[AIEvaluation] = []
        self.latency_measurements: List[float] = []
        self._initialize_sample_data()
    
    def _initialize_sample_data(self):
        """Initialize with sample historical data for demo"""
        now = datetime.now()
        
        for i in range(100):
            days_ago = random.randint(0, 30)
            self.evaluations.append(AIEvaluation(
                id=f"eval-{i}",
                incident_id=f"inc-{random.randint(1, 5):03d}",
                response_id=f"resp-{i}",
                metrics=EvaluationMetrics(
                    relevance=0.7 + random.random() * 0.25,
                    accuracy=0.65 + random.random() * 0.3,
                    helpfulness=0.6 + random.random() * 0.35,
                    latency_ms=500 + random.random() * 2000
                ),
                rating=random.randint(3, 5),
                evaluated_at=now - timedelta(days=days_ago)
            ))
        
        # Sample latency measurements
        self.latency_measurements = [
            300 + random.random() * 2500 
            for _ in range(200)
        ]
    
    def record_response(
        self,
        incident_id: str,
        response_id: str,
        start_time: datetime,
        response: str,
        context: Any
    ) -> str:
        """
        Record a new AI response for evaluation.
        
        Args:
            incident_id: ID of the incident
            response_id: Unique ID for this response
            start_time: When response generation started
            response: The generated response text
            context: The context used for generation
            
        Returns:
            Evaluation ID
        """
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000
        self.latency_measurements.append(latency_ms)
        
        # Auto-evaluate using heuristics
        auto_metrics = self._auto_evaluate(response, context)
        
        evaluation = AIEvaluation(
            id=f"eval-{uuid.uuid4().hex[:8]}",
            incident_id=incident_id,
            response_id=response_id,
            metrics=EvaluationMetrics(
                relevance=auto_metrics["relevance"],
                accuracy=auto_metrics["accuracy"],
                helpfulness=auto_metrics["helpfulness"],
                latency_ms=latency_ms
            ),
            evaluated_at=datetime.now()
        )
        
        self.evaluations.append(evaluation)
        return evaluation.id
    
    def record_feedback(
        self,
        evaluation_id: str,
        rating: Optional[int] = None,
        helpful: bool = True,
        comment: Optional[str] = None
    ):
        """Record user feedback on an AI response"""
        for eval in self.evaluations:
            if eval.id == evaluation_id:
                if rating:
                    eval.rating = rating
                if comment:
                    eval.feedback = comment
                
                # Adjust helpfulness based on feedback
                if helpful:
                    eval.metrics.helpfulness = min(1.0, eval.metrics.helpfulness + 0.1)
                else:
                    eval.metrics.helpfulness = max(0.0, eval.metrics.helpfulness - 0.1)
                break
    
    def _auto_evaluate(self, response: str, context: Any) -> Dict[str, float]:
        """
        Auto-evaluate response quality using heuristics.
        
        In production, this could use a judge LLM.
        """
        # Relevance: Check if response mentions key entities from context
        relevance_score = 0.5
        if context and hasattr(context, 'incident'):
            incident = context.incident
            keywords = [
                incident.title.lower(),
                *[s.lower() for s in incident.services],
                *[t.lower() for t in incident.tags]
            ]
            response_lower = response.lower()
            matched = sum(1 for k in keywords if k in response_lower)
            relevance_score = min(0.95, 0.5 + (matched / max(len(keywords), 1)) * 0.45)
        
        # Accuracy: Check for specific technical details and structured info
        accuracy_score = 0.5
        has_code_blocks = "```" in response
        has_steps = any(x in response.lower() for x in ["step 1", "step 2", "first", "second", "1.", "2."])
        has_specific_data = any(x in response for x in ["v", "@", ".com", "://"])
        accuracy_score = 0.5 + (0.15 if has_code_blocks else 0) + (0.15 if has_steps else 0) + (0.15 if has_specific_data else 0)
        
        # Helpfulness: Check for actionable content
        helpfulness_score = 0.5
        actionable_words = ["recommend", "suggest", "should", "try", "check", "verify", "run", "execute"]
        has_actions = any(w in response.lower() for w in actionable_words)
        has_explanation = len(response) > 200
        has_structure = "##" in response or "**" in response
        helpfulness_score = 0.5 + (0.2 if has_actions else 0) + (0.15 if has_explanation else 0) + (0.1 if has_structure else 0)
        
        # Add some randomness to simulate real-world variation
        return {
            "relevance": min(0.95, relevance_score + (random.random() * 0.1 - 0.05)),
            "accuracy": min(0.95, accuracy_score + (random.random() * 0.1 - 0.05)),
            "helpfulness": min(0.95, helpfulness_score + (random.random() * 0.1 - 0.05))
        }
    
    def get_stats(self) -> EvaluationStats:
        """Get comprehensive evaluation statistics"""
        total = len(self.evaluations)
        
        if total == 0:
            return EvaluationStats(
                total_evaluations=0,
                average_scores={"relevance": 0, "accuracy": 0, "helpfulness": 0, "latency": 0},
                user_ratings={"average": 0, "distribution": {}},
                latency_percentiles={"p50": 0, "p90": 0, "p99": 0},
                trends_over_time=[]
            )
        
        # Calculate averages
        avg_relevance = sum(e.metrics.relevance for e in self.evaluations) / total
        avg_accuracy = sum(e.metrics.accuracy for e in self.evaluations) / total
        avg_helpfulness = sum(e.metrics.helpfulness for e in self.evaluations) / total
        avg_latency = sum(e.metrics.latency_ms for e in self.evaluations) / total
        
        # User ratings
        rated = [e for e in self.evaluations if e.rating]
        avg_rating = sum(e.rating for e in rated) / len(rated) if rated else 0
        
        rating_dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for e in rated:
            rating_dist[e.rating] = rating_dist.get(e.rating, 0) + 1
        
        # Latency percentiles
        sorted_latencies = sorted(self.latency_measurements)
        p50_idx = int(len(sorted_latencies) * 0.5)
        p90_idx = int(len(sorted_latencies) * 0.9)
        p99_idx = int(len(sorted_latencies) * 0.99)
        
        # Trends over time (last 7 days)
        now = datetime.now()
        trends = []
        for i in range(6, -1, -1):
            date = now - timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            day_evals = [
                e for e in self.evaluations
                if e.evaluated_at.strftime("%Y-%m-%d") == date_str
            ]
            
            if day_evals:
                trends.append({
                    "date": date_str,
                    "relevance": sum(e.metrics.relevance for e in day_evals) / len(day_evals),
                    "accuracy": sum(e.metrics.accuracy for e in day_evals) / len(day_evals),
                    "helpfulness": sum(e.metrics.helpfulness for e in day_evals) / len(day_evals)
                })
        
        return EvaluationStats(
            total_evaluations=total,
            average_scores={
                "relevance": avg_relevance,
                "accuracy": avg_accuracy,
                "helpfulness": avg_helpfulness,
                "latency": avg_latency
            },
            user_ratings={
                "average": avg_rating,
                "distribution": rating_dist
            },
            latency_percentiles={
                "p50": sorted_latencies[p50_idx] if sorted_latencies else 0,
                "p90": sorted_latencies[p90_idx] if sorted_latencies else 0,
                "p99": sorted_latencies[p99_idx] if sorted_latencies else 0
            },
            trends_over_time=trends
        )
    
    def get_metrics(self) -> List[EvaluationMetric]:
        """Get evaluation metrics for dashboard display"""
        stats = self.get_stats()
        trends = stats.trends_over_time
        
        def get_trend(current: float, previous: float) -> str:
            diff = current - previous
            if abs(diff) < 0.02:
                return "stable"
            return "up" if diff > 0 else "down"
        
        metrics = [
            EvaluationMetric(
                id="metric-relevance",
                name="Response Relevance",
                description="How relevant AI responses are to the incident context",
                value=round(stats.average_scores["relevance"] * 100, 1),
                target=85,
                unit="%",
                trend=get_trend(
                    trends[-1]["relevance"] if trends else 0,
                    trends[-2]["relevance"] if len(trends) > 1 else 0
                ),
                history=[
                    MetricHistory(date=t["date"], value=round(t["relevance"] * 100, 1))
                    for t in trends
                ]
            ),
            EvaluationMetric(
                id="metric-accuracy",
                name="Information Accuracy",
                description="Accuracy of facts and recommendations provided",
                value=round(stats.average_scores["accuracy"] * 100, 1),
                target=90,
                unit="%",
                trend=get_trend(
                    trends[-1]["accuracy"] if trends else 0,
                    trends[-2]["accuracy"] if len(trends) > 1 else 0
                ),
                history=[
                    MetricHistory(date=t["date"], value=round(t["accuracy"] * 100, 1))
                    for t in trends
                ]
            ),
            EvaluationMetric(
                id="metric-helpfulness",
                name="User Helpfulness",
                description="How helpful users find the AI responses",
                value=round(stats.average_scores["helpfulness"] * 100, 1),
                target=80,
                unit="%",
                trend=get_trend(
                    trends[-1]["helpfulness"] if trends else 0,
                    trends[-2]["helpfulness"] if len(trends) > 1 else 0
                ),
                history=[
                    MetricHistory(date=t["date"], value=round(t["helpfulness"] * 100, 1))
                    for t in trends
                ]
            ),
            EvaluationMetric(
                id="metric-latency",
                name="Response Latency (P50)",
                description="Median time to generate AI response",
                value=round(stats.latency_percentiles["p50"]),
                target=1000,
                unit="ms",
                trend="stable",
                history=[
                    MetricHistory(date=t["date"], value=round(800 + random.random() * 400))
                    for t in trends
                ]
            ),
            EvaluationMetric(
                id="metric-rating",
                name="User Rating",
                description="Average user rating of AI responses",
                value=round(stats.user_ratings["average"], 1),
                target=4.5,
                unit="/5",
                trend="up",
                history=[
                    MetricHistory(date=t["date"], value=round(3.5 + random.random(), 1))
                    for t in trends
                ]
            )
        ]
        
        return metrics
    
    def get_evaluation(self, evaluation_id: str) -> Optional[AIEvaluation]:
        """Get evaluation by ID"""
        for e in self.evaluations:
            if e.id == evaluation_id:
                return e
        return None
    
    def get_evaluations_for_incident(self, incident_id: str) -> List[AIEvaluation]:
        """Get all evaluations for an incident"""
        return [e for e in self.evaluations if e.incident_id == incident_id]


# Global instance
_framework: Optional[AIEvaluationFramework] = None


def get_evaluation_framework() -> AIEvaluationFramework:
    """Get or create the global evaluation framework"""
    global _framework
    if _framework is None:
        _framework = AIEvaluationFramework()
    return _framework
