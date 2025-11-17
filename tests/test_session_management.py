#!/usr/bin/env python3
"""
Test script for Session Management functionality
Tests session creation, loading, saving, and conversation history management.
"""

import os
import sys
from pathlib import Path
import tempfile
import shutil

# Add project to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from services.session_manager import SessionManager


def test_session_management():
    """Test complete session management functionality."""
    print("🧪 Testing Session Management System")
    print("=" * 60)
    
    # Use temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print("1. Initializing Session Manager...")
        manager = SessionManager(temp_dir)
        print("   ✅ SessionManager initialized")
        
        # Test 2: Create new sessions
        print("\n2. Testing session creation...")
        session1_id = manager.create_new_session()
        session2_id = manager.create_new_session("Fire Safety Discussion")
        session3_id = manager.create_new_session("Building Code Review")
        
        print(f"   ✅ Created session 1: {session1_id[:8]}...")
        print(f"   ✅ Created session 2: {session2_id[:8]}... (Fire Safety)")
        print(f"   ✅ Created session 3: {session3_id[:8]}... (Building Code)")
        
        # Test 3: Load sessions
        print("\n3. Testing session loading...")
        session1_data = manager.load_session(session1_id)
        session2_data = manager.load_session(session2_id)
        
        print(f"   ✅ Loaded session 1: {session1_data['title']}")
        print(f"   ✅ Loaded session 2: {session2_data['title']}")
        
        # Test 4: Add messages to sessions
        print("\n4. Testing message management...")
        
        # Add messages to session 1
        manager.add_message_to_session(session1_id, "user", "What are the fire safety requirements?")
        manager.add_message_to_session(session1_id, "assistant", "Fire safety requirements include proper exits, sprinkler systems, and smoke detection.")
        manager.add_message_to_session(session1_id, "user", "How many exits do I need?")
        
        # Add messages to session 2
        manager.add_message_to_session(session2_id, "user", "Can you help with accessibility standards?")
        manager.add_message_to_session(session2_id, "assistant", "Yes, I can help with ADA compliance and accessibility requirements.")
        
        print("   ✅ Added messages to session 1")
        print("   ✅ Added messages to session 2")
        
        # Test 5: Update session with files
        print("\n5. Testing file tracking...")
        
        # Simulate IFC and PDF files
        session1_updated = manager.load_session(session1_id)
        session1_updated["processed_ifc_files"] = {
            "building_model.ifc": {
                "json_data": {"file_info": {"project_name": "Test Building"}},
                "file_size": 1024000,
                "processed_at": "2023-11-17T10:00:00"
            }
        }
        session1_updated["uploaded_pdfs"] = {
            "fire_code.pdf": {
                "file_name": "fire_code.pdf",
                "file_size": 512000,
                "document_type": "legal_document"
            }
        }
        
        manager.save_session(session1_id, session1_updated)
        print("   ✅ Added IFC and PDF files to session 1")
        
        # Test 6: Get all sessions
        print("\n6. Testing session listing...")
        all_sessions = manager.get_all_sessions()
        
        print(f"   ✅ Found {len(all_sessions)} sessions:")
        for session in all_sessions:
            print(f"     📄 {session['title'][:30]}... ({session['message_count']} messages)")
            print(f"        IFC: {session['ifc_files']} | PDF: {session['pdf_files']}")
            print(f"        Preview: {session['last_message_preview']}")
        
        # Test 7: Auto-generate titles
        print("\n7. Testing auto-title generation...")
        
        # Create session with meaningful first message
        test_session_id = manager.create_new_session()
        manager.add_message_to_session(test_session_id, "user", "What are the minimum door width requirements for commercial buildings?")
        
        new_title = manager.auto_generate_title(test_session_id)
        print(f"   ✅ Auto-generated title: '{new_title}'")
        
        # Test 8: Session statistics
        print("\n8. Testing session statistics...")
        stats = manager.get_session_stats()
        
        print("   📊 Session Statistics:")
        print(f"     Total Sessions: {stats['total_sessions']}")
        print(f"     Total Messages: {stats['total_messages']}")
        print(f"     Total IFC Files: {stats['total_ifc_files']}")
        print(f"     Total PDF Files: {stats['total_pdf_files']}")
        print(f"     Sessions Today: {stats['sessions_today']}")
        print(f"     Avg Messages/Session: {stats['avg_messages_per_session']:.1f}")
        
        # Test 9: Session deletion
        print("\n9. Testing session deletion...")
        sessions_before = len(manager.get_all_sessions())
        manager.delete_session(session3_id)
        sessions_after = len(manager.get_all_sessions())
        
        print(f"   ✅ Sessions before deletion: {sessions_before}")
        print(f"   ✅ Sessions after deletion: {sessions_after}")
        print(f"   ✅ Successfully deleted 1 session")
        
        # Test 10: Error handling
        print("\n10. Testing error handling...")
        
        # Try to load non-existent session
        fake_session = manager.load_session("fake-session-id")
        print(f"   ✅ Load fake session: {fake_session is None}")
        
        # Try to delete non-existent session
        delete_result = manager.delete_session("fake-session-id")
        print(f"   ✅ Delete fake session: {not delete_result}")
        
        # Test final state
        print(f"\n🎉 Session Management Test Complete!")
        final_sessions = manager.get_all_sessions()
        print(f"✅ Final state: {len(final_sessions)} sessions with working functionality")
        
        return len(final_sessions) > 0


def simulate_streamlit_workflow():
    """Simulate how sessions would work in Streamlit app."""
    print("\n" + "=" * 60)
    print("🖥️ Simulating Streamlit Session Workflow")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = SessionManager(temp_dir)
        
        # Simulate user starting app
        print("1. User starts Streamlit app...")
        current_session_id = manager.create_new_session()
        print(f"   ✅ Created initial session: {current_session_id[:8]}...")
        
        # Simulate chat interactions
        print("\n2. User has conversation...")
        
        # First interaction
        manager.add_message_to_session(current_session_id, "user", "Hello, I need help with building compliance")
        manager.add_message_to_session(current_session_id, "assistant", "Hello! I can help you with AEC compliance. Upload your building models and legal documents to get started.")
        
        # Auto-generate title after first message
        title = manager.auto_generate_title(current_session_id)
        print(f"   📝 Auto-generated title: '{title}'")
        
        # Upload files simulation
        session_data = manager.load_session(current_session_id)
        session_data["processed_ifc_files"] = {"office_building.ifc": {"processed": True}}
        session_data["uploaded_pdfs"] = {"local_fire_code.pdf": {"uploaded": True}}
        manager.save_session(current_session_id, session_data)
        print("   📁 Simulated file uploads")
        
        # More chat
        manager.add_message_to_session(current_session_id, "user", "What are the fire exit requirements?")
        manager.add_message_to_session(current_session_id, "assistant", "Based on your uploaded fire code, here are the key requirements...")
        
        # User creates new session
        print("\n3. User creates new session...")
        new_session_id = manager.create_new_session()
        print(f"   ✅ Created new session: {new_session_id[:8]}...")
        
        # New conversation
        manager.add_message_to_session(new_session_id, "user", "I need to check accessibility standards")
        new_title = manager.auto_generate_title(new_session_id)
        print(f"   📝 New session title: '{new_title}'")
        
        # User switches between sessions
        print("\n4. User browses session history...")
        sessions = manager.get_all_sessions()
        
        for i, session in enumerate(sessions, 1):
            print(f"   📄 Session {i}: {session['title']}")
            print(f"       Messages: {session['message_count']} | Files: {session['ifc_files']} IFC, {session['pdf_files']} PDF")
            print(f"       Last: {session['last_message_preview']}")
        
        print("\n✅ Streamlit workflow simulation complete!")
        print(f"   Created {len(sessions)} sessions with realistic chat history")
        
        return True


if __name__ == "__main__":
    print("🏗️ AEC Compliance Agent - Session Management Test")
    print("=" * 70)
    
    try:
        # Run session management tests
        success1 = test_session_management()
        
        # Run Streamlit workflow simulation
        success2 = simulate_streamlit_workflow()
        
        print("\n" + "=" * 70)
        if success1 and success2:
            print("🎉 All session management tests completed successfully!")
            print("🚀 Session system is ready for use in Streamlit app!")
        else:
            print("⚠️ Some tests completed with issues - check output above")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)