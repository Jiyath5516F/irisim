"""Headless OpenGL 4.5 *compatibility profile* context via GLX (run under Xvfb).

EGL on Mesa only offers core profiles for desktop GL, but Iris packs are
written against `#version 450 compatibility`, so we create the context with
glXCreateContextAttribsARB(COMPATIBILITY_PROFILE_BIT) on a 16x16 pbuffer and
render everything into our own framebuffer objects.
"""
import ctypes
import os

GLX_X_RENDERABLE = 0x8012
GLX_DRAWABLE_TYPE = 0x8010
GLX_RENDER_TYPE = 0x8011
GLX_PBUFFER_BIT = 0x0004
GLX_RGBA_BIT = 0x0001
GLX_RED_SIZE, GLX_GREEN_SIZE, GLX_BLUE_SIZE, GLX_ALPHA_SIZE = 8, 9, 10, 11
GLX_DEPTH_SIZE = 12
GLX_PBUFFER_WIDTH = 0x8041
GLX_PBUFFER_HEIGHT = 0x8040
GLX_CONTEXT_MAJOR_VERSION_ARB = 0x2091
GLX_CONTEXT_MINOR_VERSION_ARB = 0x2092
GLX_CONTEXT_FLAGS_ARB = 0x2094
GLX_CONTEXT_PROFILE_MASK_ARB = 0x9126
GLX_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB = 0x0002
GLX_CONTEXT_DEBUG_BIT_ARB = 0x0001


class GLContextError(RuntimeError):
    pass


class GLXContext:
    def __init__(self, major=4, minor=5, debug=False):
        if not os.environ.get("DISPLAY"):
            raise GLContextError("DISPLAY is not set — run irisim without --xvfb never, or under xvfb-run")
        self.x11 = ctypes.cdll.LoadLibrary("libX11.so.6")
        self.libgl = ctypes.cdll.LoadLibrary("libGL.so.1")
        x11, gl = self.x11, self.libgl
        x11.XOpenDisplay.restype = ctypes.c_void_p
        x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        x11.XDefaultScreen.argtypes = [ctypes.c_void_p]
        x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
        gl.glXChooseFBConfig.restype = ctypes.POINTER(ctypes.c_void_p)
        gl.glXChooseFBConfig.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int),
                                         ctypes.POINTER(ctypes.c_int)]
        gl.glXGetProcAddressARB.restype = ctypes.c_void_p
        gl.glXGetProcAddressARB.argtypes = [ctypes.c_char_p]
        gl.glXCreatePbuffer.restype = ctypes.c_ulong
        gl.glXCreatePbuffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        gl.glXMakeContextCurrent.restype = ctypes.c_int
        gl.glXMakeContextCurrent.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_void_p]
        gl.glXDestroyContext.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        gl.glXDestroyPbuffer.argtypes = [ctypes.c_void_p, ctypes.c_ulong]

        self.dpy = x11.XOpenDisplay(None)
        if not self.dpy:
            raise GLContextError("XOpenDisplay failed (is Xvfb running?)")
        screen = x11.XDefaultScreen(self.dpy)
        attrs = [GLX_X_RENDERABLE, 1, GLX_DRAWABLE_TYPE, GLX_PBUFFER_BIT, GLX_RENDER_TYPE, GLX_RGBA_BIT,
                 GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, GLX_ALPHA_SIZE, 0, GLX_DEPTH_SIZE, 0, 0]
        arr = (ctypes.c_int * len(attrs))(*attrs)
        n = ctypes.c_int(0)
        configs = gl.glXChooseFBConfig(self.dpy, screen, arr, ctypes.byref(n))
        if not configs or n.value == 0:
            raise GLContextError("glXChooseFBConfig found no pbuffer-capable RGBA config")
        self.fbconfig = configs[0]

        proc = gl.glXGetProcAddressARB(b"glXCreateContextAttribsARB")
        if not proc:
            raise GLContextError("glXCreateContextAttribsARB not available")
        proto = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                 ctypes.POINTER(ctypes.c_int))
        create = proto(proc)
        cattrs = [GLX_CONTEXT_MAJOR_VERSION_ARB, major, GLX_CONTEXT_MINOR_VERSION_ARB, minor,
                  GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB]
        if debug:
            cattrs += [GLX_CONTEXT_FLAGS_ARB, GLX_CONTEXT_DEBUG_BIT_ARB]
        cattrs.append(0)
        carr = (ctypes.c_int * len(cattrs))(*cattrs)
        self.ctx = create(self.dpy, self.fbconfig, None, 1, carr)
        if not self.ctx:
            raise GLContextError(f"could not create a GL {major}.{minor} compatibility context")
        parr = (ctypes.c_int * 5)(GLX_PBUFFER_WIDTH, 16, GLX_PBUFFER_HEIGHT, 16, 0)
        self.pbuffer = gl.glXCreatePbuffer(self.dpy, self.fbconfig, parr)
        if not gl.glXMakeContextCurrent(self.dpy, self.pbuffer, self.pbuffer, self.ctx):
            raise GLContextError("glXMakeContextCurrent failed")

    def release(self):
        if getattr(self, "ctx", None):
            self.libgl.glXMakeContextCurrent(self.dpy, 0, 0, None)
            self.libgl.glXDestroyPbuffer(self.dpy, self.pbuffer)
            self.libgl.glXDestroyContext(self.dpy, self.ctx)
            self.ctx = None
