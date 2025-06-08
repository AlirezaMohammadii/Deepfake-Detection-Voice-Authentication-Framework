"""
User Manager for Bayesian Networks
Handles user profiles, session management, and privacy compliance
"""

import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import logging
from collections import defaultdict
import numpy as np

@dataclass
class UserSession:
    """Session data for a user"""
    session_id: str
    user_id: str
    start_time: float
    last_activity: float
    sample_count: int = 0
    session_authenticity_scores: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class UserProfile:
    """User profile for personalized Bayesian analysis"""
    user_id: str
    created_at: float
    last_updated: float
    total_samples: int = 0
    baseline_features: Dict[str, float] = field(default_factory=dict)
    authenticity_history: List[float] = field(default_factory=list)
    risk_assessment: str = "medium"  # low, medium, high
    voice_characteristics: Dict[str, float] = field(default_factory=dict)
    adaptation_parameters: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class UserManager:
    """
    Manages user profiles and sessions for Bayesian Networks
    Handles privacy compliance and data retention
    """
    
    def __init__(self, data_dir: Optional[str] = None, enable_persistence: bool = True):
        self.logger = logging.getLogger(__name__)
        
        # Data storage setup
        if data_dir is None:
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent.parent
            data_dir = project_root / "data" / "user_profiles"
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.enable_persistence = enable_persistence
        
        # In-memory storage
        self.user_profiles: Dict[str, UserProfile] = {}
        self.active_sessions: Dict[str, UserSession] = {}
        self.session_history: Dict[str, List[UserSession]] = defaultdict(list)
        
        # Configuration
        self.max_history_length = 100
        self.session_timeout = 3600  # 1 hour
        self.data_retention_days = 30
        self.adaptation_rate = 0.1
        
        # Load existing data
        if self.enable_persistence:
            self._load_user_data()
    
    def create_user_profile(self, user_id: str, initial_metadata: Optional[Dict] = None) -> UserProfile:
        """
        Create a new user profile
        
        Args:
            user_id: Unique user identifier
            initial_metadata: Optional initial metadata
            
        Returns:
            Created UserProfile
        """
        if user_id in self.user_profiles:
            self.logger.warning(f"User profile already exists for {user_id}")
            return self.user_profiles[user_id]
        
        current_time = time.time()
        profile = UserProfile(
            user_id=user_id,
            created_at=current_time,
            last_updated=current_time,
            metadata=initial_metadata or {}
        )
        
        self.user_profiles[user_id] = profile
        
        if self.enable_persistence:
            self._save_user_profile(profile)
        
        self.logger.info(f"Created user profile for {user_id}")
        return profile
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile by ID"""
        return self.user_profiles.get(user_id)
    
    def update_user_profile(self, user_id: str, 
                          authenticity_score: Optional[float] = None,
                          features: Optional[Dict[str, float]] = None,
                          metadata_updates: Optional[Dict] = None) -> Optional[UserProfile]:
        """
        Update user profile with new data
        
        Args:
            user_id: User identifier
            authenticity_score: New authenticity score
            features: New feature values
            metadata_updates: Metadata updates
            
        Returns:
            Updated UserProfile or None if user not found
        """
        profile = self.user_profiles.get(user_id)
        if not profile:
            self.logger.warning(f"User profile not found for {user_id}")
            return None
        
        current_time = time.time()
        profile.last_updated = current_time
        profile.total_samples += 1
        
        # Update authenticity history
        if authenticity_score is not None:
            profile.authenticity_history.append(authenticity_score)
            if len(profile.authenticity_history) > self.max_history_length:
                profile.authenticity_history = profile.authenticity_history[-self.max_history_length:]
            
            # Update risk assessment
            profile.risk_assessment = self._assess_user_risk(profile.authenticity_history)
        
        # Update baseline features with exponential moving average
        if features:
            alpha = self.adaptation_rate
            for feat_name, feat_value in features.items():
                if feat_name in profile.baseline_features:
                    profile.baseline_features[feat_name] = (
                        alpha * feat_value + (1 - alpha) * profile.baseline_features[feat_name]
                    )
                else:
                    profile.baseline_features[feat_name] = feat_value
            
            # Update voice characteristics
            self._update_voice_characteristics(profile, features)
        
        # Update metadata
        if metadata_updates:
            if not hasattr(profile, 'metadata'):
                profile.metadata = {}
            profile.metadata.update(metadata_updates)
        
        if self.enable_persistence:
            self._save_user_profile(profile)
        
        return profile
    
    def start_session(self, user_id: str, session_metadata: Optional[Dict] = None) -> UserSession:
        """
        Start a new session for user
        
        Args:
            user_id: User identifier
            session_metadata: Optional session metadata
            
        Returns:
            Created UserSession
        """
        # Create user profile if it doesn't exist
        if user_id not in self.user_profiles:
            self.create_user_profile(user_id)
        
        # Generate session ID
        session_id = self._generate_session_id(user_id)
        
        current_time = time.time()
        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            start_time=current_time,
            last_activity=current_time,
            metadata=session_metadata or {}
        )
        
        self.active_sessions[session_id] = session
        
        self.logger.info(f"Started session {session_id} for user {user_id}")
        return session
    
    def update_session(self, session_id: str, 
                      authenticity_score: Optional[float] = None,
                      metadata_updates: Optional[Dict] = None) -> Optional[UserSession]:
        """
        Update active session
        
        Args:
            session_id: Session identifier
            authenticity_score: New authenticity score for this sample
            metadata_updates: Session metadata updates
            
        Returns:
            Updated UserSession or None if not found
        """
        session = self.active_sessions.get(session_id)
        if not session:
            self.logger.warning(f"Active session not found: {session_id}")
            return None
        
        current_time = time.time()
        session.last_activity = current_time
        session.sample_count += 1
        
        if authenticity_score is not None:
            session.session_authenticity_scores.append(authenticity_score)
        
        if metadata_updates:
            session.metadata.update(metadata_updates)
        
        return session
    
    def end_session(self, session_id: str) -> Optional[UserSession]:
        """
        End an active session and archive it
        
        Args:
            session_id: Session identifier
            
        Returns:
            Ended UserSession or None if not found
        """
        session = self.active_sessions.pop(session_id, None)
        if not session:
            self.logger.warning(f"Active session not found: {session_id}")
            return None
        
        # Archive session
        self.session_history[session.user_id].append(session)
        
        # Limit session history
        max_sessions = 20
        if len(self.session_history[session.user_id]) > max_sessions:
            self.session_history[session.user_id] = self.session_history[session.user_id][-max_sessions:]
        
        self.logger.info(f"Ended session {session_id} for user {session.user_id}")
        return session
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session in self.active_sessions.items():
            if current_time - session.last_activity > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.end_session(session_id)
            
        if expired_sessions:
            self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive statistics for user"""
        profile = self.user_profiles.get(user_id)
        if not profile:
            return {}
        
        # Active session info
        active_sessions = [s for s in self.active_sessions.values() if s.user_id == user_id]
        
        # Historical sessions
        historical_sessions = self.session_history.get(user_id, [])
        
        # Calculate statistics
        avg_authenticity = 0.0
        if profile.authenticity_history:
            avg_authenticity = sum(profile.authenticity_history) / len(profile.authenticity_history)
        
        return {
            'user_id': user_id,
            'profile_created': profile.created_at,
            'last_updated': profile.last_updated,
            'total_samples': profile.total_samples,
            'risk_assessment': profile.risk_assessment,
            'average_authenticity': avg_authenticity,
            'authenticity_trend': self._calculate_authenticity_trend(profile.authenticity_history),
            'active_sessions': len(active_sessions),
            'total_sessions': len(historical_sessions),
            'voice_characteristics': profile.voice_characteristics,
            'baseline_features': profile.baseline_features
        }
    
    def delete_user_data(self, user_id: str, reason: str = "user_request"):
        """
        Delete all user data for privacy compliance
        
        Args:
            user_id: User identifier
            reason: Reason for deletion
        """
        # Remove from memory
        if user_id in self.user_profiles:
            del self.user_profiles[user_id]
        
        # Remove active sessions
        sessions_to_remove = [sid for sid, session in self.active_sessions.items() 
                             if session.user_id == user_id]
        for session_id in sessions_to_remove:
            del self.active_sessions[session_id]
        
        # Remove session history
        if user_id in self.session_history:
            del self.session_history[user_id]
        
        # Remove persistent data
        if self.enable_persistence:
            user_file = self.data_dir / f"{user_id}.json"
            if user_file.exists():
                user_file.unlink()
        
        self.logger.info(f"Deleted all data for user {user_id}, reason: {reason}")
    
    def _assess_user_risk(self, authenticity_history: List[float]) -> str:
        """Assess user risk based on authenticity history"""
        if len(authenticity_history) < 3:
            return "medium"  # Insufficient data
        
        recent_scores = authenticity_history[-10:]  # Last 10 scores
        avg_score = sum(recent_scores) / len(recent_scores)
        
        if avg_score > 0.8:
            return "low"
        elif avg_score > 0.4:
            return "medium"
        else:
            return "high"
    
    def _update_voice_characteristics(self, profile: UserProfile, features: Dict[str, float]):
        """Update voice characteristics based on features"""
        # Map physics features to voice characteristics
        if 'delta_fr_revised' in features:
            profile.voice_characteristics['rotational_dynamics'] = features['delta_fr_revised']
        if 'delta_ft_revised' in features:
            profile.voice_characteristics['translational_dynamics'] = features['delta_ft_revised']
        if 'delta_fv_revised' in features:
            profile.voice_characteristics['vibrational_dynamics'] = features['delta_fv_revised']
        
        # Calculate voice distinctiveness with safe math
        if len(profile.voice_characteristics) >= 3:
            try:
                values = list(profile.voice_characteristics.values())
                # Use numpy for safer variance calculation
                variance = float(np.var(values))
                # Clamp to reasonable range to prevent overflow
                variance = min(variance, 1e6)
                profile.voice_characteristics['distinctiveness'] = variance
            except (OverflowError, ValueError) as e:
                self.logger.warning(f"Voice characteristics calculation failed: {e}")
                profile.voice_characteristics['distinctiveness'] = 0.0
    
    def _calculate_authenticity_trend(self, history: List[float]) -> str:
        """Calculate trend in authenticity scores"""
        if len(history) < 5:
            return "insufficient_data"
        
        recent = history[-5:]
        older = history[-10:-5] if len(history) >= 10 else history[:-5]
        
        if not older:
            return "insufficient_data"
        
        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)
        
        diff = recent_avg - older_avg
        
        if diff > 0.1:
            return "improving"
        elif diff < -0.1:
            return "degrading"
        else:
            return "stable"
    
    def _generate_session_id(self, user_id: str) -> str:
        """Generate unique session ID"""
        timestamp = str(time.time())
        data = f"{user_id}_{timestamp}"
        return hashlib.md5(data.encode()).hexdigest()[:16]
    
    def _save_user_profile(self, profile: UserProfile):
        """Save user profile to disk"""
        try:
            user_file = self.data_dir / f"{profile.user_id}.json"
            with open(user_file, 'w') as f:
                json.dump(asdict(profile), f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save user profile {profile.user_id}: {e}")
    
    def _load_user_data(self):
        """Load existing user data from disk"""
        try:
            for user_file in self.data_dir.glob("*.json"):
                with open(user_file, 'r') as f:
                    data = json.load(f)
                    profile = UserProfile(**data)
                    self.user_profiles[profile.user_id] = profile
                    
            self.logger.info(f"Loaded {len(self.user_profiles)} user profiles")
            
        except Exception as e:
            self.logger.error(f"Failed to load user data: {e}")
    
    def export_user_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Export all user data for data portability"""
        profile = self.user_profiles.get(user_id)
        if not profile:
            return None
        
        # Get sessions
        active_sessions = [asdict(s) for s in self.active_sessions.values() if s.user_id == user_id]
        historical_sessions = [asdict(s) for s in self.session_history.get(user_id, [])]
        
        return {
            'profile': asdict(profile),
            'active_sessions': active_sessions,
            'historical_sessions': historical_sessions,
            'statistics': self.get_user_statistics(user_id),
            'export_timestamp': time.time()
        }
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get system-wide statistics"""
        total_users = len(self.user_profiles)
        active_sessions_count = len(self.active_sessions)
        
        # Risk distribution
        risk_distribution = {"low": 0, "medium": 0, "high": 0}
        for profile in self.user_profiles.values():
            risk_distribution[profile.risk_assessment] += 1
        
        return {
            'total_users': total_users,
            'active_sessions': active_sessions_count,
            'risk_distribution': risk_distribution,
            'data_retention_days': self.data_retention_days,
            'persistence_enabled': self.enable_persistence
        } 