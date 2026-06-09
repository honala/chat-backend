from server import app

from chat_bridge import router as chat_bridge_router


app.include_router(chat_bridge_router)
