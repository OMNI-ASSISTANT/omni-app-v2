from elevenlabs import ElevenLabs
from elevenlabs.types.conversation_initiation_client_data_request_input import ConversationInitiationClientDataRequestInput


def call_user(goal: str, user: str, phone_number: str):
    client = ElevenLabs(
        api_key="sk_cf430628198af45d71bb18b6e6b9fa8e0a5b3b2c790467ee",
    )

    client.conversational_ai.twilio.outbound_call(
        agent_id="LJ7aDuf9TaSPch9MsBic",
        agent_phone_number_id="BkXlFb0SSzu1FRwIzIWs",
        to_number=phone_number,
        conversation_initiation_client_data=ConversationInitiationClientDataRequestInput(
            dynamic_variables={"goal": goal, "user": user}
        ),
    )