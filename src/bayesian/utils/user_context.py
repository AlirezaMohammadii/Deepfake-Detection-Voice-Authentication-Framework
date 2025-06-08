"""
User Context Management for Bayesian Networks
Manages user contexts and sessions for personalized analysis
"""

import time
import uuid
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class UserRole(Enum):
    """User role enumeration"""
    REGULAR = "regular"
    ADMIN = "admin"
    ANALYST = "analyst"
    GUEST = "guest"

class SessionStatus(Enum):
    """Session status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    EXPIRED = "expired"
    TERMINATED = "terminated"

@dataclass
class UserContext:
    """User context information"""
    user_id: str
    role: UserRole = UserRole.REGULAR
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    preferences: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    permissions: Set[str] = field(default_factory=set)
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_active = time.time()
    
    def is_expired(self, max_age_seconds: float = 3600.0) -> bool:
        """Check if user context is expired"""
        return (time.time() - self.last_active) > max_age_seconds

@dataclass
class SessionContext:
    """Session context information"""
    session_id: str
    user_id: str
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    session_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_active = time.time()
    
    def is_expired(self, max_age_seconds: float = 1800.0) -> bool:
        """Check if session is expired"""
        return (time.time() - self.last_active) > max_age_seconds

class UserContextManager:
    """
    Manages user contexts and sessions for Bayesian analysis
    
    Provides functionality for:
    - User context management
    - Session tracking
    - Permission management
    - Activity monitoring
    """
    
    def __init__(self, 
                 max_user_age_seconds: float = 3600.0,
                 max_session_age_seconds: float = 1800.0,
                 enable_cleanup: bool = True):
        """
        Initialize user context manager
        
        Args:
            max_user_age_seconds: Maximum age for user contexts
            max_session_age_seconds: Maximum age for sessions
            enable_cleanup: Whether to enable automatic cleanup
        """
        self.max_user_age = max_user_age_seconds
        self.max_session_age = max_session_age_seconds
        self.enable_cleanup = enable_cleanup
        
        # Storage
        self.users: Dict[str, UserContext] = {}
        self.sessions: Dict[str, SessionContext] = {}
        self.user_sessions: Dict[str, Set[str]] = {}  # user_id -> session_ids
        
        # Statistics
        self.stats = {
            'total_users': 0,
            'active_users': 0,
            'total_sessions': 0,
            'active_sessions': 0,
            'cleanup_runs': 0
        }
        
        logger.info(f"UserContextManager initialized: user_age={max_user_age_seconds}s, session_age={max_session_age_seconds}s")
    
    def create_user(self, 
                   user_id: str,
                   role: UserRole = UserRole.REGULAR,
                   preferences: Optional[Dict[str, Any]] = None,
                   metadata: Optional[Dict[str, Any]] = None,
                   permissions: Optional[Set[str]] = None) -> UserContext:
        """
        Create a new user context
        
        Args:
            user_id: Unique user identifier
            role: User role
            preferences: User preferences
            metadata: Additional metadata
            permissions: User permissions
            
        Returns:
            Created UserContext
        """
        if user_id in self.users:
            logger.warning(f"User {user_id} already exists, updating context")
            user_context = self.users[user_id]
            user_context.update_activity()
            return user_context
        
        user_context = UserContext(
            user_id=user_id,
            role=role,
            preferences=preferences or {},
            metadata=metadata or {},
            permissions=permissions or set()
        )
        
        self.users[user_id] = user_context
        self.user_sessions[user_id] = set()
        self.stats['total_users'] += 1
        self.stats['active_users'] += 1
        
        logger.info(f"Created user context: {user_id} (role: {role.value})")
        return user_context
    
    def get_user(self, user_id: str) -> Optional[UserContext]:
        """
        Get user context by ID
        
        Args:
            user_id: User identifier
            
        Returns:
            UserContext if found, None otherwise
        """
        if user_id not in self.users:
            return None
        
        user_context = self.users[user_id]
        
        # Check if expired
        if user_context.is_expired(self.max_user_age):
            logger.info(f"User context expired: {user_id}")
            self.remove_user(user_id)
            return None
        
        user_context.update_activity()
        return user_context
    
    def create_session(self, 
                      user_id: str,
                      session_data: Optional[Dict[str, Any]] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> Optional[SessionContext]:
        """
        Create a new session for a user
        
        Args:
            user_id: User identifier
            session_data: Session-specific data
            metadata: Additional metadata
            
        Returns:
            Created SessionContext or None if user not found
        """
        user_context = self.get_user(user_id)
        if not user_context:
            logger.warning(f"Cannot create session: user {user_id} not found")
            return None
        
        session_id = str(uuid.uuid4())
        
        session_context = SessionContext(
            session_id=session_id,
            user_id=user_id,
            session_data=session_data or {},
            metadata=metadata or {}
        )
        
        self.sessions[session_id] = session_context
        self.user_sessions[user_id].add(session_id)
        self.stats['total_sessions'] += 1
        self.stats['active_sessions'] += 1
        
        logger.info(f"Created session: {session_id} for user {user_id}")
        return session_context
    
    def get_session(self, session_id: str) -> Optional[SessionContext]:
        """
        Get session context by ID
        
        Args:
            session_id: Session identifier
            
        Returns:
            SessionContext if found and valid, None otherwise
        """
        if session_id not in self.sessions:
            return None
        
        session_context = self.sessions[session_id]
        
        # Check if expired
        if session_context.is_expired(self.max_session_age):
            logger.info(f"Session expired: {session_id}")
            self.end_session(session_id)
            return None
        
        session_context.update_activity()
        return session_context
    
    def end_session(self, session_id: str) -> bool:
        """
        End a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was found and ended
        """
        if session_id not in self.sessions:
            return False
        
        session_context = self.sessions[session_id]
        user_id = session_context.user_id
        
        # Update status
        session_context.status = SessionStatus.TERMINATED
        
        # Remove from active tracking
        del self.sessions[session_id]
        if user_id in self.user_sessions:
            self.user_sessions[user_id].discard(session_id)
        
        self.stats['active_sessions'] -= 1
        
        logger.info(f"Ended session: {session_id}")
        return True
    
    def remove_user(self, user_id: str) -> bool:
        """
        Remove a user and all associated sessions
        
        Args:
            user_id: User identifier
            
        Returns:
            True if user was found and removed
        """
        if user_id not in self.users:
            return False
        
        # End all user sessions
        if user_id in self.user_sessions:
            session_ids = list(self.user_sessions[user_id])
            for session_id in session_ids:
                self.end_session(session_id)
            del self.user_sessions[user_id]
        
        # Remove user
        del self.users[user_id]
        self.stats['active_users'] -= 1
        
        logger.info(f"Removed user: {user_id}")
        return True
    
    def get_user_sessions(self, user_id: str) -> List[SessionContext]:
        """
        Get all active sessions for a user
        
        Args:
            user_id: User identifier
            
        Returns:
            List of active SessionContext objects
        """
        if user_id not in self.user_sessions:
            return []
        
        active_sessions = []
        session_ids = list(self.user_sessions[user_id])
        
        for session_id in session_ids:
            session_context = self.get_session(session_id)
            if session_context:
                active_sessions.append(session_context)
        
        return active_sessions
    
    def update_user_preferences(self, 
                               user_id: str, 
                               preferences: Dict[str, Any]) -> bool:
        """
        Update user preferences
        
        Args:
            user_id: User identifier
            preferences: New preferences to merge
            
        Returns:
            True if user was found and updated
        """
        user_context = self.get_user(user_id)
        if not user_context:
            return False
        
        user_context.preferences.update(preferences)
        user_context.update_activity()
        
        logger.info(f"Updated preferences for user: {user_id}")
        return True
    
    def update_session_data(self, 
                           session_id: str, 
                           session_data: Dict[str, Any]) -> bool:
        """
        Update session data
        
        Args:
            session_id: Session identifier
            session_data: New session data to merge
            
        Returns:
            True if session was found and updated
        """
        session_context = self.get_session(session_id)
        if not session_context:
            return False
        
        session_context.session_data.update(session_data)
        session_context.update_activity()
        
        logger.info(f"Updated session data: {session_id}")
        return True
    
    def check_permission(self, user_id: str, permission: str) -> bool:
        """
        Check if user has a specific permission
        
        Args:
            user_id: User identifier
            permission: Permission to check
            
        Returns:
            True if user has permission
        """
        user_context = self.get_user(user_id)
        if not user_context:
            return False
        
        # Admin role has all permissions
        if user_context.role == UserRole.ADMIN:
            return True
        
        return permission in user_context.permissions
    
    def grant_permission(self, user_id: str, permission: str) -> bool:
        """
        Grant permission to user
        
        Args:
            user_id: User identifier
            permission: Permission to grant
            
        Returns:
            True if user was found and permission granted
        """
        user_context = self.get_user(user_id)
        if not user_context:
            return False
        
        user_context.permissions.add(permission)
        user_context.update_activity()
        
        logger.info(f"Granted permission '{permission}' to user: {user_id}")
        return True
    
    def revoke_permission(self, user_id: str, permission: str) -> bool:
        """
        Revoke permission from user
        
        Args:
            user_id: User identifier
            permission: Permission to revoke
            
        Returns:
            True if user was found and permission revoked
        """
        user_context = self.get_user(user_id)
        if not user_context:
            return False
        
        user_context.permissions.discard(permission)
        user_context.update_activity()
        
        logger.info(f"Revoked permission '{permission}' from user: {user_id}")
        return True
    
    def cleanup_expired(self) -> Dict[str, int]:
        """
        Clean up expired users and sessions
        
        Returns:
            Dictionary with cleanup statistics
        """
        if not self.enable_cleanup:
            return {'users_removed': 0, 'sessions_removed': 0}
        
        current_time = time.time()
        users_removed = 0
        sessions_removed = 0
        
        # Clean expired sessions
        expired_sessions = []
        for session_id, session_context in self.sessions.items():
            if session_context.is_expired(self.max_session_age):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.end_session(session_id)
            sessions_removed += 1
        
        # Clean expired users
        expired_users = []
        for user_id, user_context in self.users.items():
            if user_context.is_expired(self.max_user_age):
                expired_users.append(user_id)
        
        for user_id in expired_users:
            self.remove_user(user_id)
            users_removed += 1
        
        self.stats['cleanup_runs'] += 1
        
        if users_removed > 0 or sessions_removed > 0:
            logger.info(f"Cleanup completed: {users_removed} users, {sessions_removed} sessions removed")
        
        return {
            'users_removed': users_removed,
            'sessions_removed': sessions_removed
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        # Update active counts
        active_users = len([u for u in self.users.values() 
                          if not u.is_expired(self.max_user_age)])
        active_sessions = len([s for s in self.sessions.values() 
                             if not s.is_expired(self.max_session_age)])
        
        self.stats.update({
            'active_users': active_users,
            'active_sessions': active_sessions
        })
        
        return {
            **self.stats,
            'total_users_in_memory': len(self.users),
            'total_sessions_in_memory': len(self.sessions),
            'average_sessions_per_user': len(self.sessions) / max(len(self.users), 1),
            'cleanup_enabled': self.enable_cleanup
        }
    
    @property
    def active_sessions(self) -> Dict[str, SessionContext]:
        """Get all active sessions"""
        active = {}
        for session_id, session in self.sessions.items():
            if not session.is_expired(self.max_session_age):
                active[session_id] = session
        return active
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get system-wide statistics for compatibility"""
        stats = self.get_statistics()
        
        # Calculate risk distribution
        risk_distribution = {'low': 0, 'medium': 0, 'high': 0}
        for user in self.users.values():
            if not user.is_expired(self.max_user_age):
                # Simple risk assessment based on activity
                inactive_time = time.time() - user.last_active
                if inactive_time < 300:  # 5 minutes
                    risk_distribution['low'] += 1
                elif inactive_time < 1800:  # 30 minutes
                    risk_distribution['medium'] += 1
                else:
                    risk_distribution['high'] += 1
        
        return {
            'total_users': stats['total_users'],
            'active_sessions': stats['active_sessions'],
            'risk_distribution': risk_distribution,
            'system_status': 'operational',
            'cleanup_enabled': self.enable_cleanup,
            'uptime_stats': {
                'total_sessions_created': stats['total_sessions'],
                'cleanup_runs': stats['cleanup_runs'],
                'average_session_duration': self.max_session_age / 60.0  # in minutes
            }
        }
    
    def export_user_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Export all data for a user (GDPR compliance)
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with all user data or None if not found
        """
        user_context = self.get_user(user_id)
        if not user_context:
            return None
        
        # Get user sessions
        sessions_data = []
        for session_context in self.get_user_sessions(user_id):
            sessions_data.append({
                'session_id': session_context.session_id,
                'created_at': session_context.created_at,
                'last_active': session_context.last_active,
                'status': session_context.status.value,
                'session_data': session_context.session_data,
                'metadata': session_context.metadata
            })
        
        return {
            'user_context': {
                'user_id': user_context.user_id,
                'role': user_context.role.value,
                'created_at': user_context.created_at,
                'last_active': user_context.last_active,
                'preferences': user_context.preferences,
                'metadata': user_context.metadata,
                'permissions': list(user_context.permissions)
            },
            'sessions': sessions_data,
            'export_timestamp': time.time()
        } 