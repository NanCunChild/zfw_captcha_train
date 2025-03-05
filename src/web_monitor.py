# web_monitor.py
import os
import threading
import time
import json
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler
import torch
from tensorboard import program
import psutil
import webbrowser

class TrainingMonitor:
    def __init__(self, log_dir='runs'):
        self.log_dir = log_dir
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': [],
            'epoch': 0,
            'step': 0,
            'eta': 0,
            'gpu_usage': [],
            'memory_usage': [],
            'status': 'Initializing'
        }
        self.lock = threading.Lock()
        
    def update_metrics(self, metrics_dict):
        with self.lock:
            for key, value in metrics_dict.items():
                if key in self.metrics:
                    if isinstance(self.metrics[key], list):
                        self.metrics[key].append(value)
                    else:
                        self.metrics[key] = value
    
    def get_metrics(self):
        with self.lock:
            return dict(self.metrics)
    
    def update_hardware_stats(self):
        try:
            # Get GPU stats if available
            if torch.cuda.is_available():
                gpu_stats = []
                for i in range(torch.cuda.device_count()):
                    gpu_stats.append({
                        'id': i,
                        'name': torch.cuda.get_device_name(i),
                        'memory_used': torch.cuda.memory_allocated(i) / (1024**3),  # GB
                        'memory_total': torch.cuda.get_device_properties(i).total_memory / (1024**3),
                        'utilization': torch.cuda.utilization(i)
                    })
                
                with self.lock:
                    self.metrics['gpu_usage'] = gpu_stats
            
            # Get system memory usage
            memory = psutil.virtual_memory()
            with self.lock:
                self.metrics['memory_usage'] = {
                    'used': memory.used / (1024**3),  # GB
                    'total': memory.total / (1024**3),
                    'percent': memory.percent
                }
        except Exception as e:
            print(f"Error updating hardware stats: {e}")

class MonitorRequestHandler(BaseHTTPRequestHandler):
    def __init__(self, monitor, *args, **kwargs):
        self.monitor = monitor
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            with open(os.path.join(os.path.dirname(__file__), 'monitor_template.html'), 'r') as f:
                html = f.read()
            
            self.wfile.write(html.encode())
        
        elif self.path == '/api/metrics':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            metrics = self.monitor.get_metrics()
            self.wfile.write(json.dumps(metrics).encode())
        
        else:
            self.send_response(404)
            self.end_headers()
    
def create_handler(monitor):
    def handler_factory(*args, **kwargs):
        return MonitorRequestHandler(monitor, *args, **kwargs)
    return handler_factory

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def start_tensorboard(logdir, port=None):
    if port is None:
        port = find_free_port()
    
    # Start TensorBoard server
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', logdir, '--port', str(port), '--bind_all'])
    url = tb.launch()
    print(f"TensorBoard started at {url}")
    
    # Open browser
    webbrowser.open(url)
    
    return url

def start_monitoring_server(host='0.0.0.0', port=8080, log_dir='runs'):
    # Create monitor instance
    monitor = TrainingMonitor(log_dir=log_dir)
    
    # Start TensorBoard
    tensorboard_port = find_free_port()
    tb_url = start_tensorboard(log_dir, tensorboard_port)
    
    # Create HTTP server
    handler = create_handler(monitor)
    server = HTTPServer((host, port), handler)
    
    print(f"Monitoring server started at http://{host}:{port}")
    print(f"TensorBoard available at {tb_url}")
    
    # Start server in a separate thread
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    
    # Start hardware stats update thread
    def update_stats_loop():
        while True:
            monitor.update_hardware_stats()
            time.sleep(1)
    
    stats_thread = threading.Thread(target=update_stats_loop)
    stats_thread.daemon = True
    stats_thread.start()
    
    return monitor