#!/usr/bin/env python3
"""
独立运行realtime模块的启动脚本
"""

# 首先加载环境变量，在导入任何模块之前
from dotenv import load_dotenv
import os
load_dotenv()

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# 创建FastAPI应用
app = FastAPI(title="Realtime Voice Chat", version="1.0.0")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 导入realtime路由
from app.realtime.ws import router as realtime_router
app.include_router(realtime_router, prefix="/realtime", tags=["realtime"])

# 添加静态文件服务，用于提供HTML页面
@app.get("/")
async def serve_demo():
    """提供realtime demo页面"""
    return FileResponse("app/realtime/realtime_demo.html")

@app.get("/health")
async def health():
    return {"status": "ok", "message": "Realtime service is running"}

if __name__ == "__main__":
    print("🚀 启动Realtime语音对话服务...")
    print("📱 访问 http://localhost:8000 查看演示页面")
    print("🔗 WebSocket端点: ws://localhost:8000/realtime/ws/talk")
    print("💡 确保已安装所有依赖: pip install -r requirements.txt")
    print("🔑 确保.env文件中配置了OPENAI_API_KEY")
    
    uvicorn.run(
        "run_realtime:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )