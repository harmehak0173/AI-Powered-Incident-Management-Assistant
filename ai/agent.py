"""
AI Agent for Incident Management

This module implements the core AI capabilities:
- RAG-based context retrieval
- Insight generation
- Actionable recommendations
- Pattern detection across incidents
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import uuid

from models.schemas import (
    Incident, Playbook, Deployment, Service, Alert,
    AIInsight, AIContext, SimilarIncident, InsightType,
    Severity
)
from ai.vector_store import get_vector_store, SearchResult


@dataclass
class AgentResponse:
    """Response from the AI agent"""
    message: str
    insights: List[AIInsight]
    suggested_actions: List[str]
    context: AIContext
    confidence: float


class IncidentAIAgent:
    """
    AI Agent for incident analysis and recommendations.
    
    Uses RAG (Retrieval Augmented Generation) to:
    1. Find similar past incidents
    2. Retrieve relevant playbooks
    3. Identify recent deployments
    4. Generate actionable insights
    """
    
    def __init__(self, incidents: List[Incident], playbooks: List[Playbook],
                 deployments: List[Deployment], services: List[Service],
                 alerts: List[Alert]):
        self.incidents = {inc.id: inc for inc in incidents}
        self.playbooks = {pb.id: pb for pb in playbooks}
        self.deployments = deployments
        self.services = {svc.name.lower(): svc for svc in services}
        self.alerts = alerts
        self.vector_store = get_vector_store()
    
    async def analyze_incident(self, incident: Incident) -> AgentResponse:
        """
        Main entry point - analyze an incident and generate insights
        
        Args:
            incident: The incident to analyze
            
        Returns:
            AgentResponse with message, insights, suggested actions, and context
        """
        # Step 1: Retrieve relevant context using RAG
        context = await self._retrieve_context(incident)
        
        # Step 2: Generate insights based on context
        insights = self._generate_insights(incident, context)
        
        # Step 3: Create suggested actions
        suggested_actions = self._generate_suggested_actions(incident, context, insights)
        
        # Step 4: Compose response message
        message = self._compose_analysis_message(incident, context, insights)
        
        # Step 5: Build full AI context
        full_context = self._build_ai_context(incident, context)
        
        return AgentResponse(
            message=message,
            insights=insights,
            suggested_actions=suggested_actions,
            context=full_context,
            confidence=self._calculate_confidence(insights)
        )
    
    async def answer_question(self, incident: Incident, question: str) -> AgentResponse:
        """
        Answer a question about the incident using RAG
        
        Args:
            incident: The incident context
            question: User's question
            
        Returns:
            AgentResponse with the answer and relevant context
        """
        # Search for relevant context based on the question
        search_results = self.vector_store.search(question, limit=5)
        context = await self._retrieve_context(incident)
        
        # Generate response based on question type
        question_lower = question.lower()
        
        if any(kw in question_lower for kw in ['root cause', 'why', 'caused']):
            message = self._generate_root_cause_response(incident, context)
        elif any(kw in question_lower for kw in ['fix', 'resolve', 'how', 'solution']):
            message = self._generate_resolution_response(incident, context)
        elif any(kw in question_lower for kw in ['similar', 'before', 'past', 'previous']):
            message = self._generate_similar_incidents_response(context)
        elif any(kw in question_lower for kw in ['deploy', 'change', 'release', 'commit']):
            message = self._generate_deployment_response(context)
        elif any(kw in question_lower for kw in ['playbook', 'runbook', 'procedure']):
            message = self._generate_playbook_response(context)
        else:
            message = self._generate_generic_response(question, search_results)
        
        full_context = self._build_ai_context(incident, context)
        
        return AgentResponse(
            message=message,
            insights=[],
            suggested_actions=[],
            context=full_context,
            confidence=0.75
        )
    
    async def _retrieve_context(self, incident: Incident) -> Dict[str, Any]:
        """RAG: Retrieve relevant context from vector store and data sources"""
        
        # Find similar past incidents
        similar_results = self.vector_store.find_similar_incidents(incident.id, limit=5)
        similar_incidents = []
        for result in similar_results:
            source_id = result.document.metadata.get("source_id")
            if source_id and source_id in self.incidents:
                past_incident = self.incidents[source_id]
                similar_incidents.append(SimilarIncident(
                    incident=past_incident,
                    similarity=result.score,
                    matched_on=self._find_matching_attributes(incident, past_incident)
                ))
        
        # Find relevant playbooks
        playbook_results = self.vector_store.find_relevant_playbooks(
            incident.title,
            incident.services
        )
        relevant_playbooks = []
        for result in playbook_results:
            source_id = result.document.metadata.get("source_id")
            if source_id and source_id in self.playbooks:
                relevant_playbooks.append(self.playbooks[source_id])
        
        # Get recent deployments for affected services
        recent_deployments = [
            d for d in self.deployments
            if any(
                d.service.lower() in s.lower() or s.lower() in d.service.lower()
                for s in incident.services
            )
        ][:5]
        
        # Get related services
        related_services = [
            self.services[s.lower()]
            for s in incident.services
            if s.lower() in self.services
        ]
        
        # Get related alerts
        related_alerts = [
            a for a in self.alerts
            if any(s.lower() in a.service.lower() for s in incident.services)
        ]
        
        return {
            "similar_incidents": similar_incidents,
            "relevant_playbooks": relevant_playbooks,
            "recent_deployments": recent_deployments,
            "related_services": related_services,
            "related_alerts": related_alerts
        }
    
    def _generate_insights(self, incident: Incident, context: Dict[str, Any]) -> List[AIInsight]:
        """Generate insights based on incident and retrieved context"""
        insights = []
        
        # Insight 1: Similar incident pattern detection
        similar_incidents = context.get("similar_incidents", [])
        if similar_incidents:
            top_similar = similar_incidents[0]
            if top_similar.similarity > 0.5 and top_similar.incident.root_cause:
                insights.append(AIInsight(
                    id=f"insight-similar-{uuid.uuid4().hex[:8]}",
                    type=InsightType.SIMILAR_PATTERN,
                    title="Similar Incident Pattern Detected",
                    content=f'This incident closely matches "{top_similar.incident.title}" '
                           f'({int(top_similar.similarity * 100)}% similarity). '
                           f'The previous incident was caused by: {top_similar.incident.root_cause}. '
                           f'{f"It was resolved by: {top_similar.incident.resolution}" if top_similar.incident.resolution else ""}',
                    confidence=top_similar.similarity,
                    sources=[f"Incident {top_similar.incident.id}"],
                    created_at=datetime.now()
                ))
        
        # Insight 2: Recent deployment correlation
        recent_deployments = context.get("recent_deployments", [])
        very_recent = [
            d for d in recent_deployments
            if (incident.created_at - d.deployed_at).total_seconds() / 3600 <= 4
            and d.deployed_at < incident.created_at
        ]
        
        if very_recent:
            dep = very_recent[0]
            time_diff = self._format_time_diff(dep.deployed_at, incident.created_at)
            insights.append(AIInsight(
                id=f"insight-deploy-{uuid.uuid4().hex[:8]}",
                type=InsightType.ROOT_CAUSE,
                title="Recent Deployment Detected",
                content=f'A deployment to {dep.service} (v{dep.version}) occurred {time_diff} '
                       f'before this incident started. Commit: "{dep.commit_message}". '
                       f'Changed files: {", ".join(dep.changed_files[:3])}. '
                       f'Consider investigating this deployment as a potential root cause.',
                confidence=0.75,
                sources=[f"Deployment {dep.id}", dep.commit_sha],
                created_at=datetime.now()
            ))
        
        # Insight 3: Playbook recommendation
        playbooks = context.get("relevant_playbooks", [])
        if playbooks:
            pb = playbooks[0]
            step_titles = [s.title for s in pb.steps[:3]]
            insights.append(AIInsight(
                id=f"insight-playbook-{uuid.uuid4().hex[:8]}",
                type=InsightType.RECOMMENDATION,
                title="Relevant Playbook Available",
                content=f'The "{pb.title}" playbook may be helpful. It has been used '
                       f'{pb.usage_count} times with an average resolution time of '
                       f'{pb.avg_resolution_time} minutes. Key steps include: {", ".join(step_titles)}.',
                confidence=0.8,
                sources=[f"Playbook {pb.id}"],
                created_at=datetime.now()
            ))
        
        # Insight 4: Risk assessment
        services = context.get("related_services", [])
        critical_services = [
            s for s in services
            if len(s.dependencies) > 3 or 'payment' in s.name.lower() or 'database' in s.name.lower()
        ]
        
        if critical_services and incident.severity in [Severity.CRITICAL, Severity.HIGH]:
            service_names = [s.name for s in critical_services]
            insights.append(AIInsight(
                id=f"insight-risk-{uuid.uuid4().hex[:8]}",
                type=InsightType.RISK_ASSESSMENT,
                title="High Impact Assessment",
                content=f'This incident affects {", ".join(service_names)} which '
                       f'{"are" if len(critical_services) > 1 else "is"} critical to business operations. '
                       f'{f"Estimated {incident.affected_customers:,} customers affected." if incident.affected_customers else ""} '
                       f'Recommend escalating to senior on-call if not already done.',
                confidence=0.85,
                sources=[f"Service: {s.name}" for s in critical_services],
                created_at=datetime.now()
            ))
        
        return insights
    
    def _generate_suggested_actions(
        self,
        incident: Incident,
        context: Dict[str, Any],
        insights: List[AIInsight]
    ) -> List[str]:
        """Generate actionable suggestions"""
        actions = []
        
        # Actions based on status
        if incident.status.value == "investigating":
            actions.extend([
                "Review recent deployments to affected services",
                "Check monitoring dashboards for anomalies",
                "Gather error logs from affected services"
            ])
        
        # Actions based on playbook
        playbooks = context.get("relevant_playbooks", [])
        if playbooks:
            pb = playbooks[0]
            actions.append(f'Follow playbook: "{pb.title}"')
            if pb.steps:
                actions.append(f"First step: {pb.steps[0].title}")
        
        # Actions based on similar incidents
        similar = context.get("similar_incidents", [])
        resolved_similar = next(
            (s for s in similar if s.incident.status.value == "resolved" and s.incident.resolution),
            None
        )
        if resolved_similar:
            actions.append(
                f"Consider solution from similar incident: {resolved_similar.incident.resolution[:100]}..."
            )
        
        # Actions based on deployments
        deployments = context.get("recent_deployments", [])
        if deployments and deployments[0].rollback_available:
            dep = deployments[0]
            actions.append(f"Rollback available for {dep.service} v{dep.version}")
        
        # Actions based on severity
        if incident.severity == Severity.CRITICAL:
            actions.extend([
                "Consider creating a customer communication",
                "Ensure status page is updated"
            ])
        
        return actions[:6]
    
    def _compose_analysis_message(
        self,
        incident: Incident,
        context: Dict[str, Any],
        insights: List[AIInsight]
    ) -> str:
        """Compose a natural language analysis message"""
        lines = [
            f"## Incident Analysis: {incident.title}",
            "",
            f"**Severity:** {incident.severity.value.upper()} | **Status:** {incident.status.value}",
            f"**Affected Services:** {', '.join(incident.services)}",
            ""
        ]
        
        if insights:
            lines.append("### Key Findings\n")
            for insight in insights:
                confidence_emoji = "🟢" if insight.confidence > 0.7 else "🟡" if insight.confidence > 0.4 else "🟠"
                lines.append(f"{confidence_emoji} **{insight.title}** ({int(insight.confidence * 100)}% confidence)")
                lines.append(f"{insight.content}\n")
        
        similar = context.get("similar_incidents", [])
        if similar:
            lines.append("### Similar Past Incidents\n")
            for si in similar[:3]:
                lines.append(
                    f"- **{si.incident.title}** ({int(si.similarity * 100)}% match) - {si.incident.status.value}"
                )
            lines.append("")
        
        deployments = context.get("recent_deployments", [])
        if deployments:
            lines.append("### Recent Deployments\n")
            for dep in deployments[:3]:
                lines.append(f"- **{dep.service}** v{dep.version} - {dep.commit_message[:50]}...")
        
        return "\n".join(lines)
    
    def _generate_root_cause_response(self, incident: Incident, context: Dict[str, Any]) -> str:
        """Generate response about root cause"""
        lines = ["## Potential Root Cause Analysis\n"]
        
        # Check similar resolved incidents
        similar = context.get("similar_incidents", [])
        resolved_similar = next(
            (s for s in similar if s.incident.status.value == "resolved" and s.incident.root_cause),
            None
        )
        
        if resolved_similar:
            lines.append(f"Based on a similar past incident ({int(resolved_similar.similarity * 100)}% match):\n")
            lines.append(f"> **Previous Root Cause:** {resolved_similar.incident.root_cause}\n")
        
        # Check recent deployments
        deployments = context.get("recent_deployments", [])
        if deployments:
            lines.append("### Recent Changes to Consider\n")
            for dep in deployments[:3]:
                lines.append(f"- **{dep.service}** deployed {dep.version} by {dep.deployed_by}")
                lines.append(f"  - Commit: {dep.commit_message}")
                files = ', '.join(dep.changed_files[:3])
                lines.append(f"  - Files: {files}\n")
        
        if incident.root_cause:
            lines.append(f"### Identified Root Cause\n\n{incident.root_cause}")
        
        return "\n".join(lines)
    
    def _generate_resolution_response(self, incident: Incident, context: Dict[str, Any]) -> str:
        """Generate response about resolution"""
        lines = ["## Resolution Guidance\n"]
        
        playbooks = context.get("relevant_playbooks", [])
        if playbooks:
            pb = playbooks[0]
            lines.append(f"### Recommended Playbook: {pb.title}\n")
            for idx, step in enumerate(pb.steps, 1):
                lines.append(f"**Step {idx}: {step.title}**")
                lines.append(f"{step.description}")
                if step.commands:
                    lines.append("```bash")
                    lines.append("\n".join(step.commands))
                    lines.append("```")
                lines.append("")
        
        similar = context.get("similar_incidents", [])
        resolved_similar = next(
            (s for s in similar if s.incident.resolution),
            None
        )
        if resolved_similar:
            lines.append("### Previous Resolution\n")
            lines.append(f"A similar incident was resolved with:")
            lines.append(f"> {resolved_similar.incident.resolution}")
        
        return "\n".join(lines)
    
    def _generate_similar_incidents_response(self, context: Dict[str, Any]) -> str:
        """Generate response about similar incidents"""
        similar = context.get("similar_incidents", [])
        
        if not similar:
            return "No similar past incidents found in the database."
        
        lines = ["## Similar Past Incidents\n"]
        for idx, si in enumerate(similar, 1):
            inc = si.incident
            lines.append(f"### {idx}. {inc.title}")
            lines.append(f"- **Similarity:** {int(si.similarity * 100)}%")
            lines.append(f"- **Status:** {inc.status.value}")
            lines.append(f"- **Severity:** {inc.severity.value}")
            lines.append(f"- **Matched on:** {', '.join(si.matched_on)}")
            if inc.root_cause:
                lines.append(f"- **Root Cause:** {inc.root_cause}")
            if inc.resolution:
                lines.append(f"- **Resolution:** {inc.resolution}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _generate_deployment_response(self, context: Dict[str, Any]) -> str:
        """Generate response about deployments"""
        deployments = context.get("recent_deployments", [])
        
        if not deployments:
            return "No recent deployments found for the affected services."
        
        lines = ["## Recent Deployments\n"]
        for dep in deployments:
            lines.append(f"### {dep.service} v{dep.version}")
            lines.append(f"- **Deployed:** {dep.deployed_at.strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"- **By:** {dep.deployed_by}")
            lines.append(f"- **Commit:** `{dep.commit_sha}`")
            lines.append(f"- **Message:** {dep.commit_message}")
            lines.append("- **Changed Files:**")
            for f in dep.changed_files:
                lines.append(f"  - `{f}`")
            lines.append(f"- **Rollback Available:** {'Yes ✓' if dep.rollback_available else 'No'}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _generate_playbook_response(self, context: Dict[str, Any]) -> str:
        """Generate response about playbooks"""
        playbooks = context.get("relevant_playbooks", [])
        
        if not playbooks:
            return "No relevant playbooks found for this incident type."
        
        lines = ["## Relevant Playbooks\n"]
        for pb in playbooks:
            lines.append(f"### {pb.title}")
            lines.append(f"{pb.description}\n")
            lines.append(f"- **Used:** {pb.usage_count} times")
            lines.append(f"- **Avg Resolution:** {pb.avg_resolution_time} minutes\n")
            lines.append("**Steps:**")
            for idx, step in enumerate(pb.steps, 1):
                lines.append(f"{idx}. {step.title}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _generate_generic_response(self, question: str, search_results: List[SearchResult]) -> str:
        """Generate generic response based on search results"""
        if not search_results:
            return (
                f'I couldn\'t find specific information related to "{question}". '
                f'Try asking about root causes, similar incidents, recent deployments, or available playbooks.'
            )
        
        lines = ["## Search Results\n"]
        lines.append(f"Found {len(search_results)} relevant documents:\n")
        
        for idx, result in enumerate(search_results[:5], 1):
            doc = result.document
            lines.append(f"{idx}. **{doc.metadata.get('title', 'Unknown')}** ({doc.metadata.get('type', 'unknown')})")
            if result.highlights:
                lines.append(f"   > {result.highlights[0][:150]}...")
            lines.append("")
        
        return "\n".join(lines)
    
    def _build_ai_context(self, incident: Incident, context: Dict[str, Any]) -> AIContext:
        """Build the full AI context object"""
        return AIContext(
            incident=incident,
            similar_incidents=context.get("similar_incidents", []),
            relevant_playbooks=context.get("relevant_playbooks", []),
            recent_deployments=context.get("recent_deployments", []),
            related_alerts=context.get("related_alerts", []),
            service_info=context.get("related_services", [])
        )
    
    def _find_matching_attributes(self, incident1: Incident, incident2: Incident) -> List[str]:
        """Find which attributes match between two incidents"""
        matches = []
        
        if incident1.severity == incident2.severity:
            matches.append("severity")
        if any(s in incident2.services for s in incident1.services):
            matches.append("services")
        if any(t in incident2.tags for t in incident1.tags):
            matches.append("tags")
        
        words1 = set(incident1.title.lower().split())
        words2 = set(incident2.title.lower().split())
        if any(w in words2 for w in words1 if len(w) > 3):
            matches.append("keywords")
        
        return matches
    
    def _format_time_diff(self, dt1: datetime, dt2: datetime) -> str:
        """Format time difference between two datetimes"""
        diff = abs(dt2 - dt1)
        hours = diff.total_seconds() / 3600
        
        if hours >= 1:
            return f"{int(hours)} hour{'s' if hours > 1 else ''}"
        else:
            minutes = diff.total_seconds() / 60
            return f"{int(minutes)} minute{'s' if minutes > 1 else ''}"
    
    def _calculate_confidence(self, insights: List[AIInsight]) -> float:
        """Calculate overall confidence from insights"""
        if not insights:
            return 0.5
        avg_confidence = sum(i.confidence for i in insights) / len(insights)
        return min(avg_confidence, 0.95)


# Global agent instance
_agent: Optional[IncidentAIAgent] = None


def get_agent(incidents, playbooks, deployments, services, alerts) -> IncidentAIAgent:
    """Get or create the global agent instance"""
    global _agent
    if _agent is None:
        _agent = IncidentAIAgent(incidents, playbooks, deployments, services, alerts)
    return _agent
