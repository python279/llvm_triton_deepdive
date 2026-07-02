from .compiler import MyGPUBackend, register

backends = {"mygpu": register()}
