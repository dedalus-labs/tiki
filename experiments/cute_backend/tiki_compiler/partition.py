"""Split a captured program at native operations, stream changes, and shape changes."""

from .capture import Captured, TensorOperation
from .elementwise import lower
from .graph import Graph, Scalar, Symbol, Value
from .native import Collective, Matmul
from .program import Kernel, Program
from .schedule import Schedule


def lower_region(
    operations: tuple[TensorOperation, ...],
    values: dict[Symbol, Value],
    constants: tuple[Scalar, ...],
    required: set[Symbol],
    schedule: Schedule,
) -> Kernel:
    nodes = tuple(operation.node for operation in operations)
    produced = {node.output.name for node in nodes}
    constant_names = {name for name, _ in constants}
    names = dict.fromkeys(
        name for node in nodes for name in node.inputs if name not in produced | constant_names
    )
    graph = Graph(
        inputs=tuple(values[name] for name in names),
        constants=constants,
        nodes=nodes,
        outputs=tuple(node.output for node in nodes if node.output.name in required),
    )
    kernel = Kernel(lowered=lower(graph, schedule), stream=operations[0].stream)
    return kernel


def region_operations(
    operations: tuple[TensorOperation | Matmul | Collective, ...],
    position: int,
) -> tuple[TensorOperation, ...]:
    first = operations[position]
    region: list[TensorOperation] = []
    shape = ()
    for candidate in operations[position:]:
        if not isinstance(candidate, TensorOperation) or candidate.stream != first.stream:
            break
        if shape and candidate.output.shape not in ((), shape):
            break
        shape = candidate.output.shape or shape
        region.append(candidate)
    result = tuple(region)
    return result


def partition(captured: Captured, schedule: Schedule) -> Program:
    signature, operations = captured.signature, captured.operations
    values = {value.name: value for value in signature.inputs}
    values.update((name, Value.dense(name=name, shape=())) for name, _ in signature.constants)
    values.update((operation.output.name, operation.output) for operation in operations)
    stages: list[Matmul | Collective | Kernel] = []
    position = 0
    while position < len(operations):
        operation = operations[position]
        if isinstance(operation, (Matmul, Collective)):
            stages.append(operation)
            position += 1
            continue
        region = region_operations(operations, position)
        end = position + len(region)
        required = {value.name for value in signature.outputs}
        required.update(name for later in operations[end:] for name in later.inputs)
        stages.append(lower_region(region, values, signature.constants, required, schedule))
        position = end
    program = Program(
        inputs=signature.inputs,
        outputs=signature.outputs,
        constants=signature.constants,
        stages=tuple(stages),
    )
    return program
