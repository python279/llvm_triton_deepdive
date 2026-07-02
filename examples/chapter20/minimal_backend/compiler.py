"""最小可注册后端脚手架（第 20 章）"""

class MyGPUBackend:
    binary_ext = "mybin"

    def supports_target(self, target):
        return getattr(target, "backend", None) == "mygpu"

    def hash(self):
        return "mygpu-v0"

    def parse_options(self, opts):
        return opts

    def add_stages(self, stages, options, language):
        def make_mybin(mod, metadata):
            return b"MYBIN"

        stages["mybin"] = make_mybin

    def load_dialects(self, ctx):
        pass


def register():
    return MyGPUBackend()
