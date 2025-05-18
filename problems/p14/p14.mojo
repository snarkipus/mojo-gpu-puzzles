from sys import sizeof, argv
from testing import assert_equal
from gpu.host import DeviceContext

# ANCHOR: naive_matmul
from gpu import thread_idx, block_idx, block_dim, barrier
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb


alias TPB = 3
alias SIZE = 2
alias BLOCKS_PER_GRID = (1, 1)
alias THREADS_PER_BLOCK = (TPB, TPB)
alias dtype = DType.float32
alias layout = Layout.row_major(SIZE, SIZE)


fn naive_matmul[
    layout: Layout, size: Int
](
    out: LayoutTensor[mut=False, dtype, layout],
    a: LayoutTensor[mut=False, dtype, layout],
    b: LayoutTensor[mut=False, dtype, layout],
):
    row = block_dim.y * block_idx.y + thread_idx.y
    col = block_dim.x * block_idx.x + thread_idx.x
    # FILL ME IN (roughly 6 lines)

    # Allocate shared memory using tensor builder
    a_shared = tb[dtype]().row_major[TPB, TPB]().shared().alloc()
    b_shared = tb[dtype]().row_major[TPB, TPB]().shared().alloc()
    temp_products = tb[dtype]().row_major[SIZE, SIZE]().shared().alloc().fill(0)

    # Load Data
    if row < SIZE and col < SIZE:
        a_shared[row, col] = a[row, col]
        b_shared[row, col] = b[row, col]
    else:
        a_shared[row, col] = 0
        b_shared[row, col] = 0

    # Synchronize threads
    barrier()

    # Perform multiplication
    @parameter
    for k in range(SIZE):
        if row < SIZE and col < SIZE:
            temp_products[row, col] += a_shared[row, k] * b_shared[k, col]

    # Synchronize threads
    barrier()

    # Write result to global memory
    if row < SIZE and col < SIZE:
        out[row, col] = temp_products[row, col]


# ANCHOR_END: naive_matmul


# ANCHOR: single_block_matmul
fn single_block_matmul[
    layout: Layout, size: Int
](
    out: LayoutTensor[mut=False, dtype, layout],
    a: LayoutTensor[mut=False, dtype, layout],
    b: LayoutTensor[mut=False, dtype, layout],
):
    row = block_dim.y * block_idx.y + thread_idx.y
    col = block_dim.x * block_idx.x + thread_idx.x
    local_row = thread_idx.y
    local_col = thread_idx.x
    # FILL ME IN (roughly 12 lines)

    # Allocate shared memory using tensor builder
    a_shared = tb[dtype]().row_major[SIZE, SIZE]().shared().alloc()
    b_shared = tb[dtype]().row_major[SIZE, SIZE]().shared().alloc()

    # Load Data
    if row < SIZE and col < SIZE:
        a_shared[local_row, local_col] = a[row, col]
        b_shared[local_row, local_col] = b[row, col]

    # Synchronize threads
    barrier()

    # Perform multiplication
    if row < SIZE and col < SIZE:
        var acc: out.element_type = 0

        @parameter
        for k in range(SIZE):
            if row < SIZE and col < SIZE:
                acc += a_shared[local_row, k] * b_shared[k, local_col]

        out[row, col] = acc


# ANCHOR_END: single_block_matmul

# ANCHOR: matmul_tiled
alias SIZE_TILED = 8
alias BLOCKS_PER_GRID_TILED = (3, 3)  # each block convers 3x3 elements
alias THREADS_PER_BLOCK_TILED = (TPB, TPB)
alias layout_tiled = Layout.row_major(SIZE_TILED, SIZE_TILED)


fn matmul_tiled[
    layout: Layout, size: Int
](
    out: LayoutTensor[mut=False, dtype, layout],
    a: LayoutTensor[mut=False, dtype, layout],
    b: LayoutTensor[mut=False, dtype, layout],
):
    local_row = thread_idx.x
    local_col = thread_idx.y
    tiled_row = block_idx.x * TPB + local_row
    tiled_col = block_idx.y * TPB + local_col
    # FILL ME IN (roughly 20 lines)

    # Allocate shared memory using tensor builder
    a_shared = tb[dtype]().row_major[TPB, TPB]().shared().alloc()
    b_shared = tb[dtype]().row_major[TPB, TPB]().shared().alloc()

    # Initialize accumulator
    var acc: out.element_type = 0

    # Load Tile Data
    @parameter
    for tile in range((size + TPB - 1) // TPB):
        # Reset Shared Memory
        if local_row < TPB and local_col < TPB:
            a_shared[local_row, local_col] = 0
            b_shared[local_row, local_col] = 0

        barrier()

        # Load A Tile: A[...] fixed global row, column determined by tile
        if tiled_row < size and (tile * TPB + local_col) < size:
            a_shared[local_row, local_col] = a[
                tiled_row, tile * TPB + local_col
            ]

        # Load B Tile: B[...]^T row determined by tile, fixed global column
        if (tile * TPB + local_row) < size and tiled_col < size:
            b_shared[local_row, local_col] = b[
                tile * TPB + local_row, tiled_col
            ]

        # Synchronize threads
        barrier()

        # Perform Tile Multiplication
        if tiled_row < size and tiled_col < size:

            @parameter
            for k in range(TPB):
                acc += a_shared[local_row, k] * b_shared[k, local_col]

        # Synchronize threads
        barrier()

    # Write result to global memory
    if tiled_row < size and tiled_col < size:
        out[tiled_row, tiled_col] = acc


# ANCHOR_END: matmul_tiled


def main():
    with DeviceContext() as ctx:
        size = SIZE_TILED if argv()[1] == "--tiled" else SIZE
        out = ctx.enqueue_create_buffer[dtype](size * size).enqueue_fill(0)
        inp1 = ctx.enqueue_create_buffer[dtype](size * size).enqueue_fill(0)
        inp2 = ctx.enqueue_create_buffer[dtype](size * size).enqueue_fill(0)
        expected = ctx.enqueue_create_host_buffer[dtype](
            size * size
        ).enqueue_fill(0)
        with inp1.map_to_host() as inp1_host, inp2.map_to_host() as inp2_host:
            for row in range(size):
                for col in range(size):
                    val = row * size + col
                    # row major: placing elements row by row
                    inp1_host[row * size + col] = val
                    inp2_host[row * size + col] = Float32(2.0) * val

            # inp1 @ inp2.T
            for i in range(size):
                for j in range(size):
                    for k in range(size):
                        expected[i * size + j] += (
                            inp1_host[i * size + k] * inp2_host[k * size + j]
                        )

        out_tensor = LayoutTensor[mut=False, dtype, layout](out.unsafe_ptr())
        a_tensor = LayoutTensor[mut=False, dtype, layout](inp1.unsafe_ptr())
        b_tensor = LayoutTensor[mut=False, dtype, layout](inp2.unsafe_ptr())

        if argv()[1] == "--naive":
            ctx.enqueue_function[naive_matmul[layout, SIZE]](
                out_tensor,
                a_tensor,
                b_tensor,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )
        elif argv()[1] == "--single-block":
            ctx.enqueue_function[single_block_matmul[layout, SIZE]](
                out_tensor,
                a_tensor,
                b_tensor,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )
        elif argv()[1] == "--tiled":
            # Need to update the layout of the tensors to the tiled layout
            out_tensor_tiled = LayoutTensor[mut=False, dtype, layout_tiled](
                out.unsafe_ptr()
            )
            a_tensor_tiled = LayoutTensor[mut=False, dtype, layout_tiled](
                inp1.unsafe_ptr()
            )
            b_tensor_tiled = LayoutTensor[mut=False, dtype, layout_tiled](
                inp2.unsafe_ptr()
            )

            ctx.enqueue_function[matmul_tiled[layout_tiled, SIZE_TILED]](
                out_tensor_tiled,
                a_tensor_tiled,
                b_tensor_tiled,
                grid_dim=BLOCKS_PER_GRID_TILED,
                block_dim=THREADS_PER_BLOCK_TILED,
            )
        else:
            raise Error("Invalid argument")

        ctx.synchronize()

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for col in range(size):
                for row in range(size):
                    assert_equal(
                        out_host[col * size + row], expected[col * size + row]
                    )
