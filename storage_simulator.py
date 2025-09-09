import os, json, datetime
from dotenv import load_dotenv

load_dotenv()

try:
    from azure.storage.blob import BlobServiceClient
    AZURE_OK = True
except Exception:
    AZURE_OK = False

def save_to_blob(data, container="outputs/blob", use_azure=False):
    """
    Save `data` either to Azure Blob (if configured & available) or to local filesystem.
    The local fallback saves into `container` directory (which can be a path).
    """
    ts = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")

    if use_azure and AZURE_OK:
        try:
            conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
            if not conn_str:
                raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING not set")
            cname = os.getenv("AZURE_BLOB_CONTAINER", "optimizer-demo")
            blob_service = BlobServiceClient.from_connection_string(conn_str)
            container_client = blob_service.get_container_client(cname)
            try:
                container_client.create_container()
            except Exception:
                pass
            blob_name = f"refined_{ts}.json"
            container_client.get_blob_client(blob_name).upload_blob(json.dumps(data, indent=2), overwrite=True)
            print(f"[INFO] Uploaded to Azure Blob: {cname}/{blob_name}")
            return
        except Exception as e:
            print(f"[ERROR] Azure upload failed: {e} → falling back to local save")

    # Local fallback
    os.makedirs(container, exist_ok=True)
    fp = os.path.join(container, f"refined_{ts}.json")
    with open(fp, "w") as f:
        json.dump({"timestamp": ts, "data": data}, f, indent=2)
    print(f"[INFO] Saved locally: {fp}")