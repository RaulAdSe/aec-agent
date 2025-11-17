"""
Session Manager - Manages chat sessions like ChatGPT interface
Handles session creation, storage, loading, and conversation history management.
"""

import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional


class SessionManager:
    """Manages chat sessions and conversation history."""
    
    def __init__(self, sessions_dir: str = "data/sessions"):
        """Initialize with sessions directory."""
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
    
    def create_new_session(self, title: str = None) -> str:
        """
        Create a new chat session.
        
        Args:
            title: Optional session title, auto-generated if not provided
            
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
            "messages": [],
            "processed_ifc_files": {},
            "uploaded_pdfs": {},
            "metadata": {
                "message_count": 0,
                "total_ifc_files": 0,
                "total_pdf_files": 0
            }
        }
        
        # Save session file
        session_file = self.sessions_dir / f"{session_id}.json"
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        
        return session_id
    
    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a session by ID.
        
        Args:
            session_id: Session ID to load
            
        Returns:
            Session data dict or None if not found
        """
        session_file = self.sessions_dir / f"{session_id}.json"
        
        if not session_file.exists():
            return None
        
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    
    def save_session(self, session_id: str, session_data: Dict[str, Any]) -> bool:
        """
        Save session data.
        
        Args:
            session_id: Session ID
            session_data: Complete session data to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Update timestamp and metadata
            session_data["updated_at"] = datetime.now().isoformat()
            session_data["metadata"]["message_count"] = len(session_data.get("messages", []))
            session_data["metadata"]["total_ifc_files"] = len(session_data.get("processed_ifc_files", {}))
            session_data["metadata"]["total_pdf_files"] = len(session_data.get("uploaded_pdfs", {}))
            
            session_file = self.sessions_dir / f"{session_id}.json"
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception:
            return False
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """
        Get list of all sessions with summary info.
        
        Returns:
            List of session summaries sorted by update time (newest first)
        """
        sessions = []
        
        for session_file in self.sessions_dir.glob("*.json"):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                
                # Create summary
                summary = {
                    "session_id": session_data.get("session_id", session_file.stem),
                    "title": session_data.get("title", "Untitled Chat"),
                    "created_at": session_data.get("created_at"),
                    "updated_at": session_data.get("updated_at"),
                    "message_count": len(session_data.get("messages", [])),
                    "ifc_files": len(session_data.get("processed_ifc_files", {})),
                    "pdf_files": len(session_data.get("uploaded_pdfs", {})),
                    "last_message_preview": self._get_last_message_preview(session_data.get("messages", []))
                }
                
                sessions.append(summary)
                
            except Exception:
                continue  # Skip corrupted session files
        
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
        """
        Delete a session.
        
        Args:
            session_id: Session ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            session_file = self.sessions_dir / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()
                return True
            return False
        except Exception:
            return False
    
    def update_session_title(self, session_id: str, new_title: str) -> bool:
        """
        Update session title.
        
        Args:
            session_id: Session ID
            new_title: New title for the session
            
        Returns:
            True if successful, False otherwise
        """
        session_data = self.load_session(session_id)
        if session_data:
            session_data["title"] = new_title
            return self.save_session(session_id, session_data)
        return False
    
    def add_message_to_session(self, session_id: str, role: str, content: str) -> bool:
        """
        Add a message to a session.
        
        Args:
            session_id: Session ID
            role: Message role (user/assistant)
            content: Message content
            
        Returns:
            True if successful, False otherwise
        """
        session_data = self.load_session(session_id)
        if session_data:
            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat()
            }
            
            session_data["messages"].append(message)
            return self.save_session(session_id, session_data)
        return False
    
    def auto_generate_title(self, session_id: str) -> str:
        """
        Auto-generate a title based on the first user message.
        
        Args:
            session_id: Session ID
            
        Returns:
            Generated title
        """
        session_data = self.load_session(session_id)
        if not session_data:
            return "New Chat"
        
        messages = session_data.get("messages", [])
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        
        if user_messages:
            first_message = user_messages[0].get("content", "")
            
            # Extract key topics/questions
            if len(first_message) > 50:
                # Take first meaningful part
                title = first_message[:47] + "..."
            else:
                title = first_message
                
            # Clean up the title
            title = title.strip()
            if not title:
                title = "New Chat"
                
            # Update session with new title
            self.update_session_title(session_id, title)
            return title
        
        return "New Chat"
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get overall session statistics.
        
        Returns:
            Dict with session statistics
        """
        sessions = self.get_all_sessions()
        
        total_messages = sum(s.get("message_count", 0) for s in sessions)
        total_ifc_files = sum(s.get("ifc_files", 0) for s in sessions)
        total_pdf_files = sum(s.get("pdf_files", 0) for s in sessions)
        
        return {
            "total_sessions": len(sessions),
            "total_messages": total_messages,
            "total_ifc_files": total_ifc_files,
            "total_pdf_files": total_pdf_files,
            "sessions_today": len([s for s in sessions 
                                 if s.get("updated_at", "").startswith(datetime.now().strftime('%Y-%m-%d'))]),
            "avg_messages_per_session": total_messages / len(sessions) if sessions else 0
        }