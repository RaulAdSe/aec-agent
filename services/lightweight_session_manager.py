"""
Lightweight Session Manager - Minimal session management without heavy data storage
Only stores session metadata and message history, not file data.
"""

import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional


class LightweightSessionManager:
    """Manages chat sessions with minimal data storage."""
    
    def __init__(self, sessions_dir: str = "data/sessions"):
        """Initialize with sessions directory."""
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.max_sessions = 50  # Limit to prevent storage bloat
        self.max_messages_per_session = 100  # Limit message history
    
    def create_new_session(self, title: str = None) -> str:
        """
        Create a new chat session with minimal data.
        
        Args:
            title: Optional session title
            
        Returns:
            Session ID string
        """
        session_id = str(uuid.uuid4())
        
        if not title:
            title = f"New Chat {datetime.now().strftime('%m/%d %H:%M')}"
        
        session_data = {
            "session_id": session_id,
            "title": title,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "messages": [],  # Only store chat messages
            "file_summary": {  # Just counts, not actual data
                "ifc_files": 0,
                "pdf_files": 0,
                "file_names": []  # Just names for display
            }
        }
        
        # Cleanup old sessions if we have too many
        self._cleanup_old_sessions()
        
        # Save session file
        session_file = self.sessions_dir / f"{session_id}.json"
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        
        return session_id
    
    def _cleanup_old_sessions(self):
        """Remove oldest sessions if we exceed the limit."""
        sessions = self.get_all_sessions()
        
        if len(sessions) >= self.max_sessions:
            # Sort by updated_at and remove oldest
            sessions.sort(key=lambda x: x.get("updated_at", ""))
            oldest_sessions = sessions[:-self.max_sessions + 10]  # Keep some buffer
            
            for session in oldest_sessions:
                session_file = self.sessions_dir / f"{session['session_id']}.json"
                if session_file.exists():
                    session_file.unlink()
    
    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load a session by ID."""
        session_file = self.sessions_dir / f"{session_id}.json"
        
        if not session_file.exists():
            return None
        
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    
    def save_session_messages(self, session_id: str, messages: List[Dict[str, Any]]) -> bool:
        """
        Save only messages to session (not file data).
        
        Args:
            session_id: Session ID
            messages: List of chat messages
            
        Returns:
            True if successful
        """
        session_data = self.load_session(session_id)
        if not session_data:
            return False
        
        # Limit message history to prevent bloat
        if len(messages) > self.max_messages_per_session:
            messages = messages[-self.max_messages_per_session:]
        
        session_data["messages"] = messages
        session_data["updated_at"] = datetime.now().isoformat()
        
        # Auto-generate title from first message if needed
        if (session_data.get("title", "").startswith("New Chat") and 
            messages and messages[0].get("role") == "user"):
            first_message = messages[0].get("content", "")
            if first_message:
                title = first_message[:47] + "..." if len(first_message) > 47 else first_message
                session_data["title"] = title.strip()
        
        # Save back to file
        try:
            session_file = self.sessions_dir / f"{session_id}.json"
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False
    
    def update_file_summary(self, session_id: str, ifc_count: int, pdf_count: int, file_names: List[str] = None) -> bool:
        """
        Update file summary without storing actual file data.
        
        Args:
            session_id: Session ID
            ifc_count: Number of IFC files
            pdf_count: Number of PDF files
            file_names: Optional list of file names for display
            
        Returns:
            True if successful
        """
        session_data = self.load_session(session_id)
        if not session_data:
            return False
        
        session_data["file_summary"] = {
            "ifc_files": ifc_count,
            "pdf_files": pdf_count,
            "file_names": file_names[:10] if file_names else []  # Limit to 10 names
        }
        session_data["updated_at"] = datetime.now().isoformat()
        
        try:
            session_file = self.sessions_dir / f"{session_id}.json"
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get list of all sessions with summary info."""
        sessions = []
        
        for session_file in self.sessions_dir.glob("*.json"):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                
                messages = session_data.get("messages", [])
                file_summary = session_data.get("file_summary", {})
                
                summary = {
                    "session_id": session_data.get("session_id", session_file.stem),
                    "title": session_data.get("title", "Untitled Chat"),
                    "created_at": session_data.get("created_at"),
                    "updated_at": session_data.get("updated_at"),
                    "message_count": len(messages),
                    "ifc_files": file_summary.get("ifc_files", 0),
                    "pdf_files": file_summary.get("pdf_files", 0),
                    "last_message_preview": self._get_last_message_preview(messages)
                }
                
                sessions.append(summary)
                
            except Exception:
                continue  # Skip corrupted files
        
        # Sort by updated_at (newest first)
        sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        
        return sessions
    
    def _get_last_message_preview(self, messages: List[Dict[str, Any]]) -> str:
        """Get preview of the last user message."""
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        if user_messages:
            last_content = user_messages[-1].get("content", "")
            return last_content[:50] + "..." if len(last_content) > 50 else last_content
        return "No messages yet"
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        try:
            session_file = self.sessions_dir / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()
                return True
            return False
        except Exception:
            return False
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get lightweight session statistics."""
        sessions = self.get_all_sessions()
        
        total_messages = sum(s.get("message_count", 0) for s in sessions)
        total_ifc_files = sum(s.get("ifc_files", 0) for s in sessions)
        total_pdf_files = sum(s.get("pdf_files", 0) for s in sessions)
        
        return {
            "total_sessions": len(sessions),
            "total_messages": total_messages,
            "total_ifc_files": total_ifc_files,
            "total_pdf_files": total_pdf_files,
            "avg_messages_per_session": total_messages / len(sessions) if sessions else 0,
            "storage_info": {
                "max_sessions": self.max_sessions,
                "max_messages_per_session": self.max_messages_per_session,
                "current_sessions": len(sessions)
            }
        }


# Backward compatibility wrapper
class SessionManager(LightweightSessionManager):
    """Alias for backward compatibility."""
    pass