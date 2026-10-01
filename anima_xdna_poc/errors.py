"""Typed failures exposed by the prototype CLIs."""


class PrototypeError(RuntimeError):
    category = "prototype error"
    exit_code = 1


class DependencyUnavailable(PrototypeError):
    category = "dependency unavailable"
    exit_code = 2


class NPUUnavailable(PrototypeError):
    category = "NPU unavailable"
    exit_code = 3


class UnsupportedTensor(PrototypeError):
    category = "unsupported shape/dtype"
    exit_code = 4


class CompilationFailure(PrototypeError):
    category = "compilation failure"
    exit_code = 5


class ExecutionFailure(PrototypeError):
    category = "execution failure"
    exit_code = 6


class NumericalMismatch(PrototypeError):
    category = "numerical mismatch"
    exit_code = 7
