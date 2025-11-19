"""MinIO客户端"""
from minio import Minio
from minio.error import S3Error
from pathlib import Path
from typing import Optional
import io

from src.config import get_settings


class MinIOClient:
    """MinIO对象存储客户端"""
    
    def __init__(self):
        """初始化MinIO客户端"""
        from src.config import settings
        
        self.client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure
        )
        
        self.bucket_name = settings.minio_bucket
        
        # 确保bucket存在
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self) -> None:
        """确保bucket存在，不存在则创建"""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                print(f"Bucket '{self.bucket_name}' 已创建")
        except S3Error as e:
            print(f"检查/创建bucket失败: {e}")
    
    def upload_file(
        self,
        local_path: str,
        object_name: str,
        content_type: str = 'application/octet-stream'
    ) -> bool:
        """
        上传文件到MinIO
        
        Args:
            local_path: 本地文件路径
            object_name: 对象名称（MinIO中的路径）
            content_type: 内容类型
            
        Returns:
            是否上传成功
        """
        try:
            self.client.fput_object(
                self.bucket_name,
                object_name,
                local_path,
                content_type=content_type
            )
            print(f"文件已上传: {object_name}")
            return True
        except S3Error as e:
            print(f"上传文件失败: {e}")
            return False
    
    def download_file(
        self,
        object_name: str,
        local_path: str
    ) -> bool:
        """
        从MinIO下载文件
        
        Args:
            object_name: 对象名称
            local_path: 本地保存路径
            
        Returns:
            是否下载成功
        """
        try:
            # 确保目录存在
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            
            self.client.fget_object(
                self.bucket_name,
                object_name,
                local_path
            )
            print(f"文件已下载: {local_path}")
            return True
        except S3Error as e:
            print(f"下载文件失败: {e}")
            return False
    
    def upload_bytes(
        self,
        data: bytes,
        object_name: str,
        content_type: str = 'application/octet-stream'
    ) -> bool:
        """
        上传字节数据到MinIO
        
        Args:
            data: 字节数据
            object_name: 对象名称
            content_type: 内容类型
            
        Returns:
            是否上传成功
        """
        try:
            data_stream = io.BytesIO(data)
            self.client.put_object(
                self.bucket_name,
                object_name,
                data_stream,
                length=len(data),
                content_type=content_type
            )
            print(f"数据已上传: {object_name}")
            return True
        except S3Error as e:
            print(f"上传数据失败: {e}")
            return False
    
    def download_bytes(self, object_name: str) -> Optional[bytes]:
        """
        从MinIO下载字节数据
        
        Args:
            object_name: 对象名称
            
        Returns:
            字节数据，失败返回None
        """
        try:
            response = self.client.get_object(self.bucket_name, object_name)
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except S3Error as e:
            print(f"下载数据失败: {e}")
            return None
    
    def delete_file(self, object_name: str) -> bool:
        """
        删除MinIO中的文件
        
        Args:
            object_name: 对象名称
            
        Returns:
            是否删除成功
        """
        try:
            self.client.remove_object(self.bucket_name, object_name)
            print(f"文件已删除: {object_name}")
            return True
        except S3Error as e:
            print(f"删除文件失败: {e}")
            return False
    
    def list_objects(self, prefix: str = '') -> list:
        """
        列出MinIO中的对象
        
        Args:
            prefix: 对象名称前缀
            
        Returns:
            对象列表
        """
        try:
            objects = self.client.list_objects(
                self.bucket_name,
                prefix=prefix,
                recursive=True
            )
            return [obj.object_name for obj in objects]
        except S3Error as e:
            print(f"列出对象失败: {e}")
            return []
    
    def object_exists(self, object_name: str) -> bool:
        """
        检查对象是否存在
        
        Args:
            object_name: 对象名称
            
        Returns:
            是否存在
        """
        try:
            self.client.stat_object(self.bucket_name, object_name)
            return True
        except S3Error:
            return False
    
    def get_object_url(self, object_name: str, expires: int = 3600) -> Optional[str]:
        """
        获取对象的预签名URL
        
        Args:
            object_name: 对象名称
            expires: 过期时间（秒）
            
        Returns:
            预签名URL，失败返回None
        """
        try:
            url = self.client.presigned_get_object(
                self.bucket_name,
                object_name,
                expires=expires
            )
            return url
        except S3Error as e:
            print(f"生成URL失败: {e}")
            return None


class ModelStorage:
    """模型存储管理器"""
    
    def __init__(self):
        """初始化模型存储管理器"""
        self.minio_client = MinIOClient()
        self.model_prefix = "models/"
    
    def save_model(
        self,
        local_path: str,
        model_id: str,
        version: str = "v1.0"
    ) -> Optional[str]:
        """
        保存模型到MinIO
        
        Args:
            local_path: 本地模型文件路径
            model_id: 模型ID
            version: 版本号
            
        Returns:
            MinIO中的对象路径，失败返回None
        """
        object_name = f"{self.model_prefix}{model_id}/{version}/model.pth"
        
        if self.minio_client.upload_file(local_path, object_name):
            return object_name
        return None
    
    def load_model(
        self,
        model_id: str,
        version: str = "v1.0",
        local_path: str = "temp/model.pth"
    ) -> Optional[str]:
        """
        从MinIO加载模型
        
        Args:
            model_id: 模型ID
            version: 版本号
            local_path: 本地保存路径
            
        Returns:
            本地文件路径，失败返回None
        """
        object_name = f"{self.model_prefix}{model_id}/{version}/model.pth"
        
        if self.minio_client.download_file(object_name, local_path):
            return local_path
        return None
    
    def delete_model(self, model_id: str, version: str = "v1.0") -> bool:
        """
        删除模型
        
        Args:
            model_id: 模型ID
            version: 版本号
            
        Returns:
            是否删除成功
        """
        object_name = f"{self.model_prefix}{model_id}/{version}/model.pth"
        return self.minio_client.delete_file(object_name)
    
    def list_model_versions(self, model_id: str) -> list:
        """
        列出模型的所有版本
        
        Args:
            model_id: 模型ID
            
        Returns:
            版本列表
        """
        prefix = f"{self.model_prefix}{model_id}/"
        objects = self.minio_client.list_objects(prefix)
        
        # 提取版本号
        versions = set()
        for obj in objects:
            parts = obj.split('/')
            if len(parts) >= 3:
                versions.add(parts[2])
        
        return sorted(list(versions))
