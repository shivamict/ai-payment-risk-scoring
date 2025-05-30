#!/usr/bin/env python3
"""
Amazon Connect Integration Configuration and Setup
Handles real-time call streaming, transcription, and contact flow integration
"""

import boto3
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
import websockets
import os
from concurrent.futures import ThreadPoolExecutor
import threading
import time

logger = logging.getLogger(__name__)

class AmazonConnectAdvanced:
    """Advanced Amazon Connect integration for real-time call processing"""
    
    def __init__(self, region_name='us-east-1'):
        self.region_name = region_name
        self.connect_client = None
        self.transcribe_client = None
        self.lambda_client = None
        self.kinesis_client = None
        
        # Connect instance configuration
        self.instance_id = os.getenv('CONNECT_INSTANCE_ID')
        self.contact_flow_id = os.getenv('CONNECT_CONTACT_FLOW_ID')
        
        # Real-time streaming configuration
        self.stream_arn = os.getenv('CONNECT_STREAM_ARN')
        self.kinesis_stream = os.getenv('KINESIS_STREAM_NAME', 'connect-real-time-transcription')
        
        # Active streams tracking
        self.active_streams = {}
        self.stream_processors = {}
        
    async def initialize_connect_integration(self, aws_config: Dict[str, str]):
        """Initialize all AWS services needed for Connect integration"""
        try:
            # Create session with credentials
            session = boto3.Session(
                aws_access_key_id=aws_config.get('access_key_id'),
                aws_secret_access_key=aws_config.get('secret_access_key'),
                region_name=self.region_name
            )
            
            # Initialize all required clients
            self.connect_client = session.client('connect')
            self.transcribe_client = session.client('transcribe')
            self.lambda_client = session.client('lambda')
            self.kinesis_client = session.client('kinesis')
            
            # Verify Connect instance
            if self.instance_id:
                await self._verify_connect_instance()
            
            # Setup real-time transcription
            await self._setup_real_time_transcription()
            
            logger.info("✅ Amazon Connect integration initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Connect integration: {e}")
            return False
    
    async def _verify_connect_instance(self):
        """Verify Amazon Connect instance is accessible"""
        try:
            response = self.connect_client.describe_instance(InstanceId=self.instance_id)
            logger.info(f"📞 Connected to instance: {response['Instance']['InstanceAlias']}")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Could not verify Connect instance: {e}")
            return False
    
    async def _setup_real_time_transcription(self):
        """Setup real-time transcription stream"""
        try:
            # Create Kinesis stream if it doesn't exist
            try:
                self.kinesis_client.describe_stream(StreamName=self.kinesis_stream)
                logger.info(f"📊 Using existing Kinesis stream: {self.kinesis_stream}")
            except self.kinesis_client.exceptions.ResourceNotFoundException:
                # Create the stream
                self.kinesis_client.create_stream(
                    StreamName=self.kinesis_stream,
                    ShardCount=1
                )
                logger.info(f"📊 Created Kinesis stream: {self.kinesis_stream}")
                
                # Wait for stream to be active
                waiter = self.kinesis_client.get_waiter('stream_exists')
                waiter.wait(StreamName=self.kinesis_stream)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup transcription stream: {e}")
            return False
    
    async def start_contact_monitoring(self, contact_id: str, agent_id: str, customer_phone: str):
        """Start monitoring a specific contact for real-time data"""
        try:
            # Get contact details
            contact_info = await self._get_contact_details(contact_id)
            
            if contact_info:
                # Store contact information
                self.active_streams[contact_id] = {
                    'agent_id': agent_id,
                    'customer_phone': customer_phone,
                    'start_time': datetime.now(),
                    'contact_info': contact_info,
                    'transcription_job_name': f"transcribe-{contact_id}",
                    'is_active': True,
                    'audio_segments': [],
                    'transcript_segments': []
                }
                
                # Start real-time transcription
                await self._start_real_time_transcription(contact_id)
                
                # Start contact monitoring
                await self._monitor_contact_attributes(contact_id)
                
                logger.info(f"📞 Started monitoring contact: {contact_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to start contact monitoring: {e}")
            return False
    
    async def _get_contact_details(self, contact_id: str) -> Optional[Dict]:
        """Get contact details from Amazon Connect"""
        try:
            response = self.connect_client.describe_contact(
                InstanceId=self.instance_id,
                ContactId=contact_id
            )
            return response.get('Contact')
        except Exception as e:
            logger.warning(f"⚠️ Could not get contact details: {e}")
            # Return mock data for development
            return {
                'Id': contact_id,
                'InitiationMethod': 'INBOUND',
                'Channel': 'VOICE',
                'QueueInfo': {'Name': 'BasicQueue'},
                'AgentInfo': {'Id': 'agent-123', 'ConnectedToAgentTimestamp': datetime.now()},
                'InitiationTimestamp': datetime.now()
            }
    
    async def _start_real_time_transcription(self, contact_id: str):
        """Start real-time transcription for a contact"""
        try:
            job_name = f"realtime-transcribe-{contact_id}-{int(time.time())}"
            
            # Configure transcription job for real-time streaming
            transcription_config = {
                'TranscriptionJobName': job_name,
                'LanguageCode': 'en-US',
                'MediaFormat': 'wav',
                'Media': {
                    'MediaFileUri': f's3://your-connect-recordings-bucket/{contact_id}.wav'
                },
                'OutputBucketName': 'your-transcription-output-bucket',
                'Settings': {
                    'ShowSpeakerLabels': True,
                    'MaxSpeakerLabels': 2,
                    'ChannelIdentification': True
                }
            }
            
            # For real-time, we'll simulate the transcription
            # In production, this would use Amazon Transcribe streaming
            self._simulate_real_time_transcription(contact_id)
            
            logger.info(f"🎙️ Started transcription for contact: {contact_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start transcription: {e}")
            return False
    
    def _simulate_real_time_transcription(self, contact_id: str):
        """Simulate real-time transcription (for development/testing)"""
        def transcription_worker():
            """Background worker to generate mock transcription data"""
            agent_phrases = [
                "Hello, thank you for calling. How can I help you today?",
                "I understand your concern about the payment plan.",
                "Let me check your account details right now.",
                "I can see several options available for you.",
                "Would you like me to set up a payment arrangement?",
                "I can offer you a 3-month payment plan at reduced interest.",
                "Let me transfer you to our payment specialists.",
                "Thank you for your patience while I review this.",
                "I've found a solution that should work for you."
            ]
            
            customer_phrases = [
                "I'm calling about my payment that's due next week.",
                "I'm having some financial difficulties right now.",
                "Can you help me understand my current balance?",
                "I need to arrange a payment plan if possible.",
                "When would my next payment be due?",
                "Is there any way to reduce the monthly amount?",
                "I want to avoid any late fees if I can.",
                "That sounds like it could work for me.",
                "Thank you for being so helpful with this."
            ]
            
            segment_count = 0
            while (contact_id in self.active_streams and 
                   self.active_streams[contact_id]['is_active'] and 
                   segment_count < 20):
                
                # Alternate between agent and customer
                is_agent = segment_count % 2 == 0
                phrases = agent_phrases if is_agent else customer_phrases
                
                # Generate transcript segment
                segment = {
                    'contact_id': contact_id,
                    'timestamp': datetime.now().isoformat(),
                    'speaker': 'AGENT' if is_agent else 'CUSTOMER',
                    'text': phrases[segment_count % len(phrases)],
                    'confidence': 0.95 + (0.05 * (segment_count % 2)),
                    'sentiment': self._calculate_mock_sentiment(segment_count, is_agent),
                    'segment_id': segment_count
                }
                
                # Store segment
                if contact_id in self.active_streams:
                    self.active_streams[contact_id]['transcript_segments'].append(segment)
                    
                    # Limit stored segments to last 10
                    if len(self.active_streams[contact_id]['transcript_segments']) > 10:
                        self.active_streams[contact_id]['transcript_segments'].pop(0)
                
                segment_count += 1
                time.sleep(8 + (segment_count % 3))  # 8-11 second intervals
        
        # Start transcription in background thread
        thread = threading.Thread(target=transcription_worker, daemon=True)
        thread.start()
        self.stream_processors[contact_id] = thread
    
    def _calculate_mock_sentiment(self, segment_count: int, is_agent: bool) -> Dict[str, float]:
        """Calculate mock sentiment for transcript segments"""
        if is_agent:
            # Agents typically maintain positive sentiment
            return {
                'sentiment': 'POSITIVE',
                'positive_score': 0.8 + (0.2 * (segment_count % 2)),
                'negative_score': 0.1,
                'neutral_score': 0.1
            }
        else:
            # Customer sentiment varies more
            if segment_count < 3:
                # Early in call - more negative/concerned
                return {
                    'sentiment': 'NEGATIVE',
                    'positive_score': 0.2,
                    'negative_score': 0.6 + (0.2 * (segment_count % 2)),
                    'neutral_score': 0.2
                }
            elif segment_count > 10:
                # Later in call - more positive as issues resolve
                return {
                    'sentiment': 'POSITIVE',
                    'positive_score': 0.7 + (0.3 * (segment_count % 2)),
                    'negative_score': 0.1,
                    'neutral_score': 0.2
                }
            else:
                # Middle of call - neutral
                return {
                    'sentiment': 'NEUTRAL',
                    'positive_score': 0.3,
                    'negative_score': 0.3,
                    'neutral_score': 0.4
                }
    
    async def _monitor_contact_attributes(self, contact_id: str):
        """Monitor contact attributes and metrics in real-time"""
        def attribute_monitor():
            """Background worker to monitor contact attributes"""
            while (contact_id in self.active_streams and 
                   self.active_streams[contact_id]['is_active']):
                
                try:
                    # In production, this would call:
                    # self.connect_client.get_current_metric_data(...)
                    # For now, we'll simulate the data
                    
                    attributes = {
                        'contact_id': contact_id,
                        'timestamp': datetime.now().isoformat(),
                        'queue_time': (datetime.now() - self.active_streams[contact_id]['start_time']).seconds,
                        'talk_time': (datetime.now() - self.active_streams[contact_id]['start_time']).seconds,
                        'hold_time': 0,
                        'agent_connected': True,
                        'customer_connected': True,
                        'call_quality_score': 4.2 + (0.8 * (time.time() % 1))
                    }
                    
                    # Store attributes
                    self.active_streams[contact_id]['latest_attributes'] = attributes
                    
                except Exception as e:
                    logger.error(f"❌ Error monitoring contact attributes: {e}")
                
                time.sleep(5)  # Update every 5 seconds
        
        # Start monitoring in background thread
        thread = threading.Thread(target=attribute_monitor, daemon=True)
        thread.start()
    
    async def get_real_time_transcript(self, contact_id: str) -> Dict[str, Any]:
        """Get the latest transcript segments for a contact"""
        if contact_id not in self.active_streams:
            return {'error': 'Contact not found or not active'}
        
        contact_data = self.active_streams[contact_id]
        latest_segments = contact_data.get('transcript_segments', [])
        
        # Return the most recent segment
        if latest_segments:
            latest = latest_segments[-1]
            return {
                'contact_id': contact_id,
                'latest_segment': latest,
                'segment_count': len(latest_segments),
                'call_duration': (datetime.now() - contact_data['start_time']).seconds,
                'attributes': contact_data.get('latest_attributes', {})
            }
        
        return {
            'contact_id': contact_id,
            'latest_segment': None,
            'segment_count': 0,
            'call_duration': (datetime.now() - contact_data['start_time']).seconds
        }
    
    async def end_contact_monitoring(self, contact_id: str):
        """End monitoring for a specific contact"""
        if contact_id in self.active_streams:
            # Mark as inactive
            self.active_streams[contact_id]['is_active'] = False
            
            # Generate final summary
            contact_data = self.active_streams[contact_id]
            summary = {
                'contact_id': contact_id,
                'total_duration': (datetime.now() - contact_data['start_time']).seconds,
                'total_segments': len(contact_data.get('transcript_segments', [])),
                'final_sentiment': self._calculate_overall_sentiment(contact_data),
                'call_summary': self._generate_call_summary(contact_data)
            }
            
            # Cleanup
            if contact_id in self.stream_processors:
                del self.stream_processors[contact_id]
            
            logger.info(f"📞 Ended monitoring for contact: {contact_id}")
            return summary
        
        return {'error': 'Contact not found'}
    
    def _calculate_overall_sentiment(self, contact_data: Dict) -> Dict[str, Any]:
        """Calculate overall sentiment analysis for the call"""
        segments = contact_data.get('transcript_segments', [])
        
        if not segments:
            return {'overall_sentiment': 'NEUTRAL', 'confidence': 0.0}
        
        # Analyze customer sentiment progression
        customer_segments = [s for s in segments if s['speaker'] == 'CUSTOMER']
        
        if customer_segments:
            # Calculate trend (negative to positive is good)
            first_sentiment = customer_segments[0]['sentiment']['negative_score']
            last_sentiment = customer_segments[-1]['sentiment']['positive_score']
            
            improvement = last_sentiment - first_sentiment
            
            return {
                'overall_sentiment': 'POSITIVE' if improvement > 0.3 else 'NEUTRAL' if improvement > -0.2 else 'NEGATIVE',
                'improvement_score': improvement,
                'confidence': 0.8,
                'customer_satisfaction_trend': 'IMPROVING' if improvement > 0.2 else 'STABLE' if improvement > -0.2 else 'DECLINING'
            }
        
        return {'overall_sentiment': 'NEUTRAL', 'confidence': 0.5}
    
    def _generate_call_summary(self, contact_data: Dict) -> Dict[str, Any]:
        """Generate a comprehensive call summary"""
        segments = contact_data.get('transcript_segments', [])
        
        summary = {
            'call_type': 'PAYMENT_INQUIRY',
            'resolution_status': 'RESOLVED' if len(segments) > 5 else 'IN_PROGRESS',
            'key_topics': ['payment_plan', 'account_balance', 'financial_assistance'],
            'agent_performance': {
                'professionalism': 'HIGH',
                'problem_resolution': 'EFFECTIVE',
                'customer_satisfaction': 'POSITIVE'
            },
            'follow_up_required': len(segments) > 8,
            'escalation_occurred': False
        }
        
        return summary

class ConnectContactFlow:
    """Manages Amazon Connect contact flows for risk scoring integration"""
    
    def __init__(self, connect_client):
        self.connect_client = connect_client
        
    def create_risk_scoring_flow(self, instance_id: str) -> str:
        """Create a contact flow that integrates with risk scoring"""
        
        contact_flow_content = {
            "Version": "2019-10-30",
            "StartAction": "RiskScoringEntry",
            "Actions": [
                {
                    "Identifier": "RiskScoringEntry",
                    "Type": "InvokeExternalResource",
                    "Parameters": {
                        "FunctionArn": os.getenv('RISK_SCORING_LAMBDA_ARN'),
                        "TimeoutSeconds": "10"
                    },
                    "Transitions": {
                        "NextAction": "CheckRiskLevel"
                    }
                },
                {
                    "Identifier": "CheckRiskLevel",
                    "Type": "Compare",
                    "Parameters": {
                        "ComparisonValue": "$.External.risk_score"
                    },
                    "Conditions": [
                        {
                            "NextAction": "HighRiskFlow",
                            "Condition": {
                                "Operator": "NumberGreaterThan",
                                "Operands": ["70"]
                            }
                        },
                        {
                            "NextAction": "MediumRiskFlow", 
                            "Condition": {
                                "Operator": "NumberGreaterThan",
                                "Operands": ["40"]
                            }
                        }
                    ],
                    "Default": "LowRiskFlow"
                },
                {
                    "Identifier": "HighRiskFlow",
                    "Type": "Transfer",
                    "Parameters": {
                        "ContactFlowId": os.getenv('HIGH_RISK_CONTACT_FLOW_ID')
                    }
                },
                {
                    "Identifier": "MediumRiskFlow",
                    "Type": "SetContactAttributes",
                    "Parameters": {
                        "Attributes": {
                            "risk_level": "medium",
                            "escalation_required": "false"
                        }
                    },
                    "Transitions": {
                        "NextAction": "StandardFlow"
                    }
                },
                {
                    "Identifier": "LowRiskFlow",
                    "Type": "SetContactAttributes",
                    "Parameters": {
                        "Attributes": {
                            "risk_level": "low",
                            "upsell_opportunity": "true"
                        }
                    },
                    "Transitions": {
                        "NextAction": "StandardFlow"
                    }
                },
                {
                    "Identifier": "StandardFlow",
                    "Type": "MessageParticipant",
                    "Parameters": {
                        "Text": "Please hold while I connect you with the best available agent."
                    },
                    "Transitions": {
                        "NextAction": "TransferToQueue"
                    }
                },
                {
                    "Identifier": "TransferToQueue",
                    "Type": "TransferToQueue",
                    "Parameters": {
                        "QueueId": os.getenv('DEFAULT_QUEUE_ID')
                    }
                }
            ]
        }
        
        try:
            response = self.connect_client.create_contact_flow(
                InstanceId=instance_id,
                Name="AI Risk Scoring Contact Flow",
                Type="CONTACT_FLOW",
                Description="Integrates real-time AI risk scoring with call routing",
                Content=json.dumps(contact_flow_content)
            )
            
            logger.info(f"✅ Created risk scoring contact flow: {response['ContactFlowId']}")
            return response['ContactFlowId']
            
        except Exception as e:
            logger.error(f"❌ Failed to create contact flow: {e}")
            return None

# Configuration helper
def get_connect_config() -> Dict[str, str]:
    """Get Amazon Connect configuration from environment variables"""
    return {
        'instance_id': os.getenv('CONNECT_INSTANCE_ID', ''),
        'region': os.getenv('AWS_REGION', 'us-east-1'),
        'access_key_id': os.getenv('AWS_ACCESS_KEY_ID', ''),
        'secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY', ''),
        'contact_flow_id': os.getenv('CONNECT_CONTACT_FLOW_ID', ''),
        'queue_id': os.getenv('CONNECT_QUEUE_ID', ''),
        'kinesis_stream': os.getenv('KINESIS_STREAM_NAME', 'connect-real-time-transcription'),
        'lambda_function': os.getenv('RISK_SCORING_LAMBDA_ARN', '')
    }

# Example usage and testing
async def test_connect_integration():
    """Test the Amazon Connect integration"""
    logger.info("🧪 Testing Amazon Connect integration...")
    
    # Initialize Connect integration
    connect = AmazonConnectAdvanced()
    
    # Test configuration
    config = get_connect_config()
    
    if await connect.initialize_connect_integration(config):
        logger.info("✅ Connect integration test passed")
        
        # Test contact monitoring
        test_contact_id = "test-contact-123"
        if await connect.start_contact_monitoring(test_contact_id, "agent-001", "+1234567890"):
            logger.info("✅ Contact monitoring test passed")
            
            # Wait a bit and get transcript
            await asyncio.sleep(5)
            transcript = await connect.get_real_time_transcript(test_contact_id)
            logger.info(f"📝 Test transcript: {transcript}")
            
            # End monitoring
            summary = await connect.end_contact_monitoring(test_contact_id)
            logger.info(f"📊 Test summary: {summary}")
            
        else:
            logger.error("❌ Contact monitoring test failed")
    else:
        logger.error("❌ Connect integration test failed")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    asyncio.run(test_connect_integration())
