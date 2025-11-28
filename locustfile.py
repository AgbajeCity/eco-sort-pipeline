
from locust import HttpUser, task, between
class WasteUser(HttpUser):
    wait_time = between(0.5, 1)
    @task
    def index(self):
        self.client.get("/_stcore/health")
