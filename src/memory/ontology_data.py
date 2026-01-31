"""
Ontology Data Models and Sample Data Generators

Provides domain-specific ontology schemas and sample data generation for:
1. Customer Churn Analysis - Customer usage data
2. CI/CD Pipeline - Pipeline execution data with successes/failures
3. User Management - User access history

These ontologies integrate with Fabric IQ for AI-grounded reasoning.
"""

import uuid
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

from .facts_memory import (
    OntologyEntity, OntologyRelationship, Fact,
    EntityType, RelationshipType,
    CustomerEntity, PipelineEntity, UserAccessEntity,
)


# =============================================================================
# CUSTOMER CHURN ANALYSIS ONTOLOGY
# =============================================================================

class CustomerSegment(Enum):
    """Customer segmentation based on value and behavior"""
    ENTERPRISE = "enterprise"
    BUSINESS = "business"
    PROFESSIONAL = "professional"
    STARTER = "starter"
    TRIAL = "trial"


class ChurnRiskLevel(Enum):
    """Churn risk classification"""
    CRITICAL = "critical"  # > 80%
    HIGH = "high"          # 60-80%
    MEDIUM = "medium"      # 40-60%
    LOW = "low"            # 20-40%
    MINIMAL = "minimal"    # < 20%


@dataclass
class CustomerProfile:
    """Complete customer profile for churn analysis"""
    customer_id: str
    name: str
    email: str
    segment: CustomerSegment
    tenure_months: int
    monthly_spend: float
    total_spend: float
    
    # Engagement metrics
    login_frequency: float  # logins per week
    feature_usage_score: float  # 0-100
    support_tickets_30d: int
    nps_score: int  # -100 to 100
    
    # Churn indicators
    days_since_last_login: int
    payment_issues_count: int
    downgrade_requests: int
    
    # Calculated churn risk
    churn_risk: float  # 0-1
    risk_level: ChurnRiskLevel
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_activity_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "customer_id": self.customer_id,
            "name": self.name,
            "email": self.email,
            "segment": self.segment.value,
            "tenure_months": self.tenure_months,
            "monthly_spend": self.monthly_spend,
            "total_spend": self.total_spend,
            "login_frequency": self.login_frequency,
            "feature_usage_score": self.feature_usage_score,
            "support_tickets_30d": self.support_tickets_30d,
            "nps_score": self.nps_score,
            "days_since_last_login": self.days_since_last_login,
            "payment_issues_count": self.payment_issues_count,
            "downgrade_requests": self.downgrade_requests,
            "churn_risk": self.churn_risk,
            "risk_level": self.risk_level.value,
            "created_at": self.created_at,
            "last_activity_at": self.last_activity_at,
        }


@dataclass
class CustomerTransaction:
    """Customer transaction record"""
    transaction_id: str
    customer_id: str
    transaction_type: str  # purchase, renewal, upgrade, downgrade, refund
    amount: float
    currency: str
    product: str
    timestamp: str
    status: str  # completed, pending, failed, refunded
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EngagementEvent:
    """Customer engagement event"""
    event_id: str
    customer_id: str
    event_type: str  # login, feature_use, support_ticket, feedback, export
    feature_name: str
    duration_seconds: int
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# CI/CD PIPELINE ONTOLOGY
# =============================================================================

class PipelineStatus(Enum):
    """Pipeline run status"""
    SUCCESS = "success"
    FAILURE = "failure"
    IN_PROGRESS = "in_progress"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class FailureCategory(Enum):
    """Categories of pipeline failures"""
    BUILD_ERROR = "build_error"
    TEST_FAILURE = "test_failure"
    LINT_ERROR = "lint_error"
    SECURITY_SCAN = "security_scan"
    DEPLOYMENT_ERROR = "deployment_error"
    INFRASTRUCTURE = "infrastructure"
    TIMEOUT = "timeout"
    RESOURCE_LIMIT = "resource_limit"
    CONFIGURATION = "configuration"


@dataclass
class Pipeline:
    """CI/CD Pipeline definition"""
    pipeline_id: str
    name: str
    repository: str
    branch: str
    target_cluster: str
    service_name: str
    
    # Configuration
    stages: List[str]
    trigger_type: str  # push, pull_request, schedule, manual
    auto_deploy: bool
    
    # Statistics
    total_runs: int
    success_count: int
    failure_count: int
    avg_duration_seconds: float
    
    # Metadata
    created_at: str
    last_run_at: str
    status: str = "active"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @property
    def success_rate(self) -> float:
        if self.total_runs == 0:
            return 0.0
        return self.success_count / self.total_runs


@dataclass
class PipelineRun:
    """Individual pipeline execution"""
    run_id: str
    pipeline_id: str
    commit_sha: str
    branch: str
    trigger: str  # push, pr, schedule, manual, rollback
    triggered_by: str  # username or "system"
    
    # Execution details
    status: PipelineStatus
    started_at: str
    completed_at: str
    duration_seconds: int
    
    # Stage results
    stages_completed: List[str]
    current_stage: str
    
    # Failure info (if applicable)
    failure_category: str = None
    failure_message: str = None
    failure_stage: str = None
    
    # Deployment info
    deployed_to: str = None
    deployment_version: str = None
    rollback_of: str = None
    
    # Metrics
    tests_passed: int = 0
    tests_failed: int = 0
    coverage_percent: float = 0.0
    security_issues: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass
class DeploymentEvent:
    """Kubernetes deployment event"""
    deployment_id: str
    pipeline_run_id: str
    cluster_name: str
    namespace: str
    service_name: str
    
    # Deployment details
    image_tag: str
    replicas: int
    status: str  # pending, running, succeeded, failed, rolledback
    
    # Timing
    started_at: str
    completed_at: str
    
    # Health
    ready_replicas: int = 0
    pod_restarts: int = 0
    health_check_passed: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# USER MANAGEMENT ONTOLOGY
# =============================================================================

class UserStatus(Enum):
    """User account status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING_VERIFICATION = "pending_verification"
    LOCKED = "locked"


class AuthEventType(Enum):
    """Authentication event types"""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    PASSWORD_CHANGE = "password_change"
    PASSWORD_RESET = "password_reset"
    MFA_ENABLED = "mfa_enabled"
    MFA_DISABLED = "mfa_disabled"
    MFA_CHALLENGE = "mfa_challenge"
    TOKEN_REFRESH = "token_refresh"
    SESSION_EXPIRED = "session_expired"
    ACCOUNT_LOCKED = "account_locked"
    ACCOUNT_UNLOCKED = "account_unlocked"


class AccessAction(Enum):
    """Resource access actions"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"


@dataclass
class User:
    """User account"""
    user_id: str
    email: str
    username: str
    first_name: str
    last_name: str
    
    # Account details
    status: UserStatus
    roles: List[str]
    permissions: List[str]
    
    # Security
    mfa_enabled: bool
    last_password_change: str
    failed_login_attempts: int
    
    # Activity
    created_at: str
    last_login: str
    total_sessions: int
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass
class AuthEvent:
    """Authentication event record"""
    event_id: str
    user_id: str
    event_type: AuthEventType
    timestamp: str
    
    # Context
    ip_address: str
    user_agent: str
    location: str  # city, country
    device_type: str  # desktop, mobile, tablet, api
    
    # Session info
    session_id: str = None
    
    # Additional details
    success: bool = True
    failure_reason: str = None
    risk_score: float = 0.0  # 0-1, higher = more suspicious
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["event_type"] = self.event_type.value
        return data


@dataclass
class AccessLog:
    """Resource access log entry"""
    log_id: str
    user_id: str
    session_id: str
    timestamp: str
    
    # Access details
    resource_type: str  # api, data, file, admin
    resource_path: str
    action: AccessAction
    
    # Request info
    method: str  # GET, POST, PUT, DELETE
    status_code: int
    response_time_ms: int
    
    # Context
    ip_address: str
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["action"] = self.action.value
        return data


# =============================================================================
# SAMPLE DATA GENERATORS
# =============================================================================

class CustomerDataGenerator:
    """Generate sample customer usage data for churn analysis"""
    
    FIRST_NAMES = ["Emma", "Liam", "Olivia", "Noah", "Ava", "Ethan", "Sophia", "Mason",
                   "Isabella", "William", "Mia", "James", "Charlotte", "Benjamin", "Amelia"]
    LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
                  "Davis", "Rodriguez", "Martinez", "Anderson", "Taylor", "Thomas", "Moore"]
    PRODUCTS = ["Pro Plan", "Business Suite", "Enterprise License", "Starter Pack", 
                "Analytics Add-on", "Integration Module", "Support Premium"]
    FEATURES = ["Dashboard", "Reports", "API Access", "Integrations", "Exports",
                "Automation", "Collaboration", "Analytics", "Admin Panel"]
    
    @classmethod
    def generate_customers(cls, count: int = 50) -> List[CustomerProfile]:
        """Generate sample customer profiles."""
        customers = []
        
        for i in range(count):
            customer_id = f"cust-{uuid.uuid4().hex[:8]}"
            first_name = random.choice(cls.FIRST_NAMES)
            last_name = random.choice(cls.LAST_NAMES)
            
            # Generate realistic distributions
            segment = random.choices(
                list(CustomerSegment),
                weights=[0.05, 0.15, 0.30, 0.35, 0.15]  # More starters, fewer enterprise
            )[0]
            
            # Segment-based attributes
            segment_config = {
                CustomerSegment.ENTERPRISE: {"tenure": (24, 60), "spend": (5000, 20000)},
                CustomerSegment.BUSINESS: {"tenure": (12, 48), "spend": (1000, 5000)},
                CustomerSegment.PROFESSIONAL: {"tenure": (6, 36), "spend": (200, 1000)},
                CustomerSegment.STARTER: {"tenure": (1, 24), "spend": (50, 200)},
                CustomerSegment.TRIAL: {"tenure": (0, 1), "spend": (0, 50)},
            }
            
            config = segment_config[segment]
            tenure = random.randint(*config["tenure"])
            monthly_spend = round(random.uniform(*config["spend"]), 2)
            
            # Calculate churn indicators
            days_since_login = random.choices(
                [random.randint(0, 3), random.randint(4, 14), random.randint(15, 60)],
                weights=[0.6, 0.25, 0.15]
            )[0]
            
            login_frequency = max(0, 7 - days_since_login / 3 + random.uniform(-1, 1))
            feature_usage = max(10, min(100, 70 - days_since_login + random.uniform(-20, 20)))
            support_tickets = random.choices([0, 1, 2, 3, 4, 5], weights=[0.4, 0.25, 0.15, 0.1, 0.05, 0.05])[0]
            nps_score = random.randint(-20, 100) if days_since_login < 30 else random.randint(-100, 40)
            payment_issues = random.choices([0, 1, 2, 3], weights=[0.7, 0.15, 0.1, 0.05])[0]
            downgrade_requests = random.choices([0, 1, 2], weights=[0.85, 0.1, 0.05])[0]
            
            # Calculate churn risk
            churn_risk = cls._calculate_churn_risk(
                days_since_login, login_frequency, feature_usage,
                support_tickets, nps_score, payment_issues, downgrade_requests, tenure
            )
            
            risk_level = cls._get_risk_level(churn_risk)
            
            customer = CustomerProfile(
                customer_id=customer_id,
                name=f"{first_name} {last_name}",
                email=f"{first_name.lower()}.{last_name.lower()}@example.com",
                segment=segment,
                tenure_months=tenure,
                monthly_spend=monthly_spend,
                total_spend=round(monthly_spend * tenure, 2),
                login_frequency=round(login_frequency, 1),
                feature_usage_score=round(feature_usage, 1),
                support_tickets_30d=support_tickets,
                nps_score=nps_score,
                days_since_last_login=days_since_login,
                payment_issues_count=payment_issues,
                downgrade_requests=downgrade_requests,
                churn_risk=round(churn_risk, 3),
                risk_level=risk_level,
                created_at=(datetime.utcnow() - timedelta(days=tenure * 30)).isoformat(),
                last_activity_at=(datetime.utcnow() - timedelta(days=days_since_login)).isoformat(),
            )
            customers.append(customer)
        
        return customers
    
    @classmethod
    def generate_transactions(cls, customer: CustomerProfile, count: int = 10) -> List[CustomerTransaction]:
        """Generate transaction history for a customer."""
        transactions = []
        
        for i in range(count):
            days_ago = random.randint(0, customer.tenure_months * 30)
            timestamp = (datetime.utcnow() - timedelta(days=days_ago)).isoformat()
            
            tx_type = random.choices(
                ["purchase", "renewal", "upgrade", "downgrade", "refund"],
                weights=[0.3, 0.4, 0.15, 0.1, 0.05]
            )[0]
            
            amount = customer.monthly_spend
            if tx_type == "upgrade":
                amount *= random.uniform(1.2, 2.0)
            elif tx_type == "downgrade":
                amount *= random.uniform(0.3, 0.8)
            elif tx_type == "refund":
                amount = -abs(amount) * random.uniform(0.1, 1.0)
            
            status = random.choices(
                ["completed", "pending", "failed", "refunded"],
                weights=[0.85, 0.05, 0.05, 0.05]
            )[0]
            
            transactions.append(CustomerTransaction(
                transaction_id=f"tx-{uuid.uuid4().hex[:8]}",
                customer_id=customer.customer_id,
                transaction_type=tx_type,
                amount=round(amount, 2),
                currency="USD",
                product=random.choice(cls.PRODUCTS),
                timestamp=timestamp,
                status=status,
            ))
        
        return transactions
    
    @classmethod
    def generate_engagement_events(cls, customer: CustomerProfile, count: int = 20) -> List[EngagementEvent]:
        """Generate engagement events for a customer."""
        events = []
        
        for i in range(count):
            days_ago = random.randint(0, min(30, customer.tenure_months * 30))
            timestamp = (datetime.utcnow() - timedelta(days=days_ago)).isoformat()
            
            event_type = random.choices(
                ["login", "feature_use", "support_ticket", "feedback", "export"],
                weights=[0.35, 0.40, 0.10, 0.05, 0.10]
            )[0]
            
            events.append(EngagementEvent(
                event_id=f"evt-{uuid.uuid4().hex[:8]}",
                customer_id=customer.customer_id,
                event_type=event_type,
                feature_name=random.choice(cls.FEATURES),
                duration_seconds=random.randint(30, 3600),
                timestamp=timestamp,
                metadata={"source": "web", "version": "2.0"},
            ))
        
        return events
    
    @staticmethod
    def _calculate_churn_risk(
        days_since_login: int,
        login_frequency: float,
        feature_usage: float,
        support_tickets: int,
        nps_score: int,
        payment_issues: int,
        downgrade_requests: int,
        tenure: int,
    ) -> float:
        """Calculate churn risk score (0-1)."""
        risk = 0.0
        
        # Days since login (major factor)
        if days_since_login > 30:
            risk += 0.35
        elif days_since_login > 14:
            risk += 0.20
        elif days_since_login > 7:
            risk += 0.10
        
        # Login frequency
        if login_frequency < 1:
            risk += 0.15
        elif login_frequency < 3:
            risk += 0.08
        
        # Feature usage
        if feature_usage < 30:
            risk += 0.15
        elif feature_usage < 50:
            risk += 0.08
        
        # Support tickets (inverse - could indicate engagement or problems)
        if support_tickets > 3:
            risk += 0.10
        
        # NPS score
        if nps_score < 0:
            risk += 0.15
        elif nps_score < 30:
            risk += 0.08
        
        # Payment issues
        risk += payment_issues * 0.08
        
        # Downgrade requests
        risk += downgrade_requests * 0.12
        
        # Tenure (new customers more likely to churn)
        if tenure < 3:
            risk += 0.10
        elif tenure > 24:
            risk -= 0.05  # Long-term customers less likely
        
        return max(0.0, min(1.0, risk))
    
    @staticmethod
    def _get_risk_level(risk: float) -> ChurnRiskLevel:
        """Convert risk score to risk level."""
        if risk > 0.8:
            return ChurnRiskLevel.CRITICAL
        elif risk > 0.6:
            return ChurnRiskLevel.HIGH
        elif risk > 0.4:
            return ChurnRiskLevel.MEDIUM
        elif risk > 0.2:
            return ChurnRiskLevel.LOW
        else:
            return ChurnRiskLevel.MINIMAL


class PipelineDataGenerator:
    """Generate sample CI/CD pipeline execution data"""
    
    SERVICES = ["api-gateway", "user-service", "order-service", "notification-service",
                "payment-service", "inventory-service", "analytics-service", "auth-service"]
    CLUSTERS = ["aks-prod-eastus", "aks-prod-westeu", "aks-staging", "aks-dev"]
    STAGES = ["checkout", "build", "test", "security-scan", "deploy"]
    DEVELOPERS = ["alice", "bob", "charlie", "diana", "evan", "fiona", "george"]
    
    @classmethod
    def generate_pipelines(cls, count: int = 8) -> List[Pipeline]:
        """Generate pipeline definitions."""
        pipelines = []
        
        for i, service in enumerate(cls.SERVICES[:count]):
            pipeline_id = f"pipe-{uuid.uuid4().hex[:8]}"
            total_runs = random.randint(50, 500)
            success_count = int(total_runs * random.uniform(0.75, 0.98))
            
            pipelines.append(Pipeline(
                pipeline_id=pipeline_id,
                name=f"{service}-pipeline",
                repository=f"org/{service}",
                branch="main",
                target_cluster=random.choice(cls.CLUSTERS),
                service_name=service,
                stages=cls.STAGES.copy(),
                trigger_type=random.choice(["push", "pull_request", "schedule"]),
                auto_deploy=random.choice([True, False]),
                total_runs=total_runs,
                success_count=success_count,
                failure_count=total_runs - success_count,
                avg_duration_seconds=random.randint(180, 900),
                created_at=(datetime.utcnow() - timedelta(days=random.randint(60, 365))).isoformat(),
                last_run_at=(datetime.utcnow() - timedelta(hours=random.randint(0, 48))).isoformat(),
            ))
        
        return pipelines
    
    @classmethod
    def generate_pipeline_runs(cls, pipeline: Pipeline, count: int = 30) -> List[PipelineRun]:
        """Generate pipeline execution history."""
        runs = []
        
        for i in range(count):
            run_id = f"run-{uuid.uuid4().hex[:8]}"
            hours_ago = random.randint(0, 720)  # Up to 30 days
            started_at = (datetime.utcnow() - timedelta(hours=hours_ago))
            duration = random.randint(120, 900)
            
            # Determine success/failure
            is_success = random.random() < pipeline.success_rate
            
            if is_success:
                status = PipelineStatus.SUCCESS
                failure_category = None
                failure_message = None
                failure_stage = None
                stages_completed = pipeline.stages.copy()
                current_stage = "complete"
            else:
                status = PipelineStatus.FAILURE
                failure_stage = random.choice(pipeline.stages)
                failure_idx = pipeline.stages.index(failure_stage)
                stages_completed = pipeline.stages[:failure_idx]
                current_stage = failure_stage
                failure_category = random.choice(list(FailureCategory)).value
                
                failure_messages = {
                    FailureCategory.BUILD_ERROR.value: "Compilation failed: undefined reference to 'main'",
                    FailureCategory.TEST_FAILURE.value: "3 tests failed in UserServiceTest",
                    FailureCategory.LINT_ERROR.value: "ESLint found 12 errors",
                    FailureCategory.SECURITY_SCAN.value: "Critical vulnerability CVE-2024-1234 detected",
                    FailureCategory.DEPLOYMENT_ERROR.value: "Pod failed health check after 5 attempts",
                    FailureCategory.INFRASTRUCTURE.value: "Unable to connect to AKS cluster",
                    FailureCategory.TIMEOUT.value: "Job exceeded maximum duration of 30 minutes",
                    FailureCategory.RESOURCE_LIMIT.value: "Out of memory during build process",
                    FailureCategory.CONFIGURATION.value: "Missing required environment variable DB_HOST",
                }
                failure_message = failure_messages.get(failure_category, "Unknown error")
            
            tests_passed = random.randint(100, 500)
            tests_failed = 0 if is_success else random.randint(1, 10)
            
            runs.append(PipelineRun(
                run_id=run_id,
                pipeline_id=pipeline.pipeline_id,
                commit_sha=uuid.uuid4().hex[:7],
                branch=random.choice(["main", "develop", f"feature/{random.choice(['auth', 'api', 'fix', 'update'])}"]),
                trigger=random.choice(["push", "pr", "schedule", "manual"]),
                triggered_by=random.choice(cls.DEVELOPERS) if random.random() > 0.2 else "system",
                status=status,
                started_at=started_at.isoformat(),
                completed_at=(started_at + timedelta(seconds=duration)).isoformat(),
                duration_seconds=duration,
                stages_completed=stages_completed,
                current_stage=current_stage,
                failure_category=failure_category,
                failure_message=failure_message,
                failure_stage=failure_stage,
                deployed_to=pipeline.target_cluster if is_success else None,
                deployment_version=f"v1.{random.randint(0, 50)}.{random.randint(0, 99)}",
                tests_passed=tests_passed,
                tests_failed=tests_failed,
                coverage_percent=round(random.uniform(70, 95), 1),
                security_issues=0 if is_success else random.randint(0, 5),
            ))
        
        return runs
    
    @classmethod
    def generate_deployments(cls, pipeline_run: PipelineRun, cluster: str) -> DeploymentEvent:
        """Generate deployment event from pipeline run."""
        is_success = pipeline_run.status == PipelineStatus.SUCCESS
        
        return DeploymentEvent(
            deployment_id=f"deploy-{uuid.uuid4().hex[:8]}",
            pipeline_run_id=pipeline_run.run_id,
            cluster_name=cluster,
            namespace=random.choice(["default", "production", "staging"]),
            service_name=pipeline_run.pipeline_id.replace("pipe-", ""),
            image_tag=pipeline_run.deployment_version or "latest",
            replicas=random.randint(2, 5),
            status="succeeded" if is_success else "failed",
            started_at=pipeline_run.started_at,
            completed_at=pipeline_run.completed_at,
            ready_replicas=random.randint(2, 5) if is_success else random.randint(0, 1),
            pod_restarts=0 if is_success else random.randint(1, 10),
            health_check_passed=is_success,
        )


class UserAccessDataGenerator:
    """Generate sample user access and authentication data"""
    
    USERNAMES = ["jsmith", "mjohnson", "ewilliams", "rbrown", "kkhan", "slee", 
                 "mgarcia", "tchen", "apatil", "dkim", "admin", "sysop"]
    ROLES = ["user", "admin", "developer", "analyst", "manager", "viewer", "auditor"]
    LOCATIONS = ["New York, US", "London, UK", "Tokyo, JP", "Sydney, AU", 
                 "Berlin, DE", "Toronto, CA", "Mumbai, IN", "Singapore, SG"]
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14) Safari/17.0",
        "Mozilla/5.0 (iPhone; CPU iOS 17) Mobile Safari",
        "PostmanRuntime/7.36.0",
        "Python-urllib/3.11",
    ]
    RESOURCES = [
        "/api/v1/users", "/api/v1/orders", "/api/v1/products", "/api/v1/reports",
        "/api/v1/analytics", "/api/v1/settings", "/admin/users", "/admin/roles",
        "/data/export", "/data/import", "/files/documents", "/files/uploads"
    ]
    
    @classmethod
    def generate_users(cls, count: int = 20) -> List[User]:
        """Generate user accounts."""
        users = []
        
        for i, username in enumerate(cls.USERNAMES[:count]):
            user_id = f"user-{uuid.uuid4().hex[:8]}"
            
            # Role assignment
            if username in ["admin", "sysop"]:
                roles = ["admin", "user"]
            elif username.startswith(("dev", "eng")):
                roles = ["developer", "user"]
            else:
                roles = ["user"]
            
            # Add random secondary role
            if random.random() > 0.7:
                roles.append(random.choice(["analyst", "viewer"]))
            
            status = random.choices(
                list(UserStatus),
                weights=[0.80, 0.08, 0.05, 0.05, 0.02]
            )[0]
            
            days_since_creation = random.randint(30, 730)
            last_login_days = random.randint(0, min(30, days_since_creation))
            
            users.append(User(
                user_id=user_id,
                email=f"{username}@company.com",
                username=username,
                first_name=username.split(".")[0].capitalize() if "." in username else username.capitalize(),
                last_name="User",
                status=status,
                roles=roles,
                permissions=cls._derive_permissions(roles),
                mfa_enabled=random.choice([True, False]),
                last_password_change=(datetime.utcnow() - timedelta(days=random.randint(0, 90))).isoformat(),
                failed_login_attempts=random.choices([0, 1, 2, 3, 5], weights=[0.7, 0.15, 0.08, 0.05, 0.02])[0],
                created_at=(datetime.utcnow() - timedelta(days=days_since_creation)).isoformat(),
                last_login=(datetime.utcnow() - timedelta(days=last_login_days)).isoformat(),
                total_sessions=random.randint(10, 500),
            ))
        
        return users
    
    @classmethod
    def generate_auth_events(cls, user: User, count: int = 50) -> List[AuthEvent]:
        """Generate authentication events for a user."""
        events = []
        
        for i in range(count):
            hours_ago = random.randint(0, 720)
            timestamp = (datetime.utcnow() - timedelta(hours=hours_ago))
            
            # Event type distribution
            event_type = random.choices(
                [AuthEventType.LOGIN_SUCCESS, AuthEventType.LOGIN_FAILURE, 
                 AuthEventType.LOGOUT, AuthEventType.TOKEN_REFRESH,
                 AuthEventType.PASSWORD_CHANGE, AuthEventType.MFA_CHALLENGE],
                weights=[0.45, 0.10, 0.25, 0.15, 0.02, 0.03]
            )[0]
            
            is_success = event_type != AuthEventType.LOGIN_FAILURE
            
            # Risk score calculation
            risk_score = 0.0
            location = random.choice(cls.LOCATIONS)
            device_type = random.choices(
                ["desktop", "mobile", "tablet", "api"],
                weights=[0.5, 0.25, 0.10, 0.15]
            )[0]
            
            # Increase risk for unusual patterns
            if device_type == "api":
                risk_score += 0.1
            if "Unknown" in location:
                risk_score += 0.2
            if event_type == AuthEventType.LOGIN_FAILURE:
                risk_score += 0.3
            
            failure_reason = None
            if event_type == AuthEventType.LOGIN_FAILURE:
                failure_reason = random.choice([
                    "Invalid password",
                    "Account locked",
                    "Invalid username",
                    "MFA verification failed",
                    "Session expired",
                ])
            
            events.append(AuthEvent(
                event_id=f"auth-{uuid.uuid4().hex[:8]}",
                user_id=user.user_id,
                event_type=event_type,
                timestamp=timestamp.isoformat(),
                ip_address=f"{random.randint(1, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}",
                user_agent=random.choice(cls.USER_AGENTS),
                location=location,
                device_type=device_type,
                session_id=f"sess-{uuid.uuid4().hex[:8]}",
                success=is_success,
                failure_reason=failure_reason,
                risk_score=round(min(1.0, risk_score), 2),
            ))
        
        return events
    
    @classmethod
    def generate_access_logs(cls, user: User, session_id: str, count: int = 30) -> List[AccessLog]:
        """Generate access log entries for a user session."""
        logs = []
        
        for i in range(count):
            minutes_ago = random.randint(0, 480)
            timestamp = (datetime.utcnow() - timedelta(minutes=minutes_ago))
            
            resource = random.choice(cls.RESOURCES)
            
            # Determine action based on resource and user role
            if "/admin" in resource and "admin" not in user.roles:
                action = AccessAction.READ  # Limited access
                status_code = 403
            else:
                action = random.choices(
                    [AccessAction.READ, AccessAction.CREATE, AccessAction.UPDATE, AccessAction.DELETE],
                    weights=[0.6, 0.2, 0.15, 0.05]
                )[0]
                status_code = random.choices([200, 201, 400, 401, 403, 404, 500], weights=[0.8, 0.05, 0.05, 0.02, 0.03, 0.03, 0.02])[0]
            
            method_map = {
                AccessAction.READ: "GET",
                AccessAction.CREATE: "POST",
                AccessAction.UPDATE: "PUT",
                AccessAction.DELETE: "DELETE",
                AccessAction.EXECUTE: "POST",
                AccessAction.ADMIN: "POST",
            }
            
            logs.append(AccessLog(
                log_id=f"log-{uuid.uuid4().hex[:8]}",
                user_id=user.user_id,
                session_id=session_id,
                timestamp=timestamp.isoformat(),
                resource_type="api" if resource.startswith("/api") else "admin" if "/admin" in resource else "data",
                resource_path=resource,
                action=action,
                method=method_map.get(action, "GET"),
                status_code=status_code,
                response_time_ms=random.randint(10, 2000),
                ip_address=f"{random.randint(1, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}",
            ))
        
        return logs
    
    @staticmethod
    def _derive_permissions(roles: List[str]) -> List[str]:
        """Derive permissions from roles."""
        permissions = []
        
        role_permissions = {
            "user": ["read:own_profile", "update:own_profile", "read:public_data"],
            "admin": ["read:all_users", "update:all_users", "delete:users", "manage:roles"],
            "developer": ["read:api_docs", "create:api_keys", "access:dev_tools"],
            "analyst": ["read:reports", "export:data", "create:dashboards"],
            "manager": ["read:team_data", "approve:requests", "view:analytics"],
            "viewer": ["read:public_data"],
            "auditor": ["read:audit_logs", "export:audit_data"],
        }
        
        for role in roles:
            permissions.extend(role_permissions.get(role, []))
        
        return list(set(permissions))
