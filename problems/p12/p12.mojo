from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from sys import sizeof, argv
from math import log2
from testing import assert_equal

# ANCHOR: prefix_sum_simple
alias TPB = 8
alias SIZE = 8
alias BLOCKS_PER_GRID = (1, 1)
alias THREADS_PER_BLOCK = (TPB, 1)
alias dtype = DType.float32
alias layout = Layout.row_major(SIZE)


fn prefix_sum_simple[
    layout: Layout
](
    out: LayoutTensor[mut=False, dtype, layout],
    a: LayoutTensor[mut=False, dtype, layout],
    size: Int,
):
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x
    # FILL ME IN (roughly 12 lines)

    # Allocate shared memory using tensor builder
    shared = tb[dtype]().row_major[TPB]().shared().alloc()

    # Load Data
    if global_i < SIZE:
        shared[local_i] = a[global_i]

#   Threads:       T₀   T₁   T₂   T₃   T₄   T₅   T₆   T₇
#   Input array: [ 0    1    2    3    4    5    6    7 ]
#   shared:      [ 0    1    2    3    4    5    6    7 ]
#                  ↑    ↑    ↑    ↑    ↑    ↑    ↑    ↑
#                  T₀   T₁   T₂   T₃   T₄   T₅   T₆   T₇

    barrier()

#   Offset: +1 (T₁ .. T₇)
#   Before:      [ 0    1    2    3    4    5    6    7 ]
#   Add:               +0   +1   +2   +3   +4   +5   +6 
#                       |    |    |    |    |    |    |
#   After:       [ 0    1    3    5    7    9   11   13 ]
#                       ↑    ↑    ↑    ↑    ↑    ↑    ↑
#                       T₁   T₂   T₃   T₄   T₅   T₆   T₇
# -------------------------------------------------------
#   Offset: +2 (T₂ .. T₇)
#   Before:      [ 0    1    3    5    7    9   11   13 ]
#   Add:                    +0   +1   +3   +5   +7   +9  
#                            |    |    |    |    |    |
#   After:       [ 0    1    3    6   10   14   18   22 ]
#                            ↑    ↑    ↑    ↑    ↑    ↑
#                            T₂   T₃   T₄   T₅   T₆   T₇
# -------------------------------------------------------
#   Offset: +4 (T₄ .. T₇)
#   Before:      [ 0    1    3    6   10   14   18   22 ]
#   Add:                              +0   +1   +3   +6
#                                      |    |    |    |
#   After:       [ 0    1    3    6   10   15   21   28 ]
#                                      ↑    ↑    ↑    ↑
#                                      T₄   T₅   T₆   T₇
# -------------------------------------------------------
#   Final Output:
#   Threads:       T₀   T₁   T₂   T₃   T₄   T₅   T₆   T₇
#   global_i:      0    1    2    3    4    5    6    7 
#   out[]:       [ 0    1    3    6   10   15   21   28 ]
#                  ↑    ↑    ↑    ↑    ↑    ↑    ↑    ↑
#                  T₀   T₁   T₂   T₃   T₄   T₅   T₆   T₇

    offset = 1

    for _ in range(Int(log2(Scalar[dtype](TPB)))):
        if local_i >= offset and local_i < SIZE:
            shared[local_i] += shared[local_i - offset]

        barrier()
        offset *= 2

    if global_i < SIZE:
        out[global_i] = shared[local_i]

# ANCHOR_END: prefix_sum_simple

# ANCHOR: prefix_sum_complete
alias SIZE_2 = 15
alias BLOCKS_PER_GRID_2 = (2, 1)
alias THREADS_PER_BLOCK_2 = (TPB, 1)
alias EXTENDED_SIZE = SIZE_2 + 2  # up to 2 blocks
alias extended_layout = Layout.row_major(EXTENDED_SIZE)


# Kernel 1: Compute local prefix sums and store block sums in out
fn prefix_sum_local_phase[
    out_layout: Layout, in_layout: Layout
](
    out: LayoutTensor[mut=False, dtype, out_layout],
    a: LayoutTensor[mut=False, dtype, in_layout],
    size: Int,
    num_blocks: Int,
):
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x
    # FILL ME IN (roughly 14 lines)


# Kernel 2: Add block sums to their respective blocks
fn prefix_sum_block_sum_phase[
    layout: Layout
](out: LayoutTensor[mut=False, dtype, layout], size: Int):
    global_i = block_dim.x * block_idx.x + thread_idx.x
    # FILL ME IN (roughly 3 lines)


# ANCHOR_END: prefix_sum_complete


def main():
    with DeviceContext() as ctx:
        use_simple = argv()[1] == "--simple"
        size = SIZE if use_simple else SIZE_2
        num_blocks = (size + TPB - 1) // TPB

        if not use_simple and num_blocks > EXTENDED_SIZE - SIZE_2:
            raise Error("Extended buffer too small for the number of blocks")

        buffer_size = size if use_simple else EXTENDED_SIZE
        out = ctx.enqueue_create_buffer[dtype](buffer_size).enqueue_fill(0)
        a = ctx.enqueue_create_buffer[dtype](size).enqueue_fill(0)

        with a.map_to_host() as a_host:
            for i in range(size):
                a_host[i] = i

        a_tensor = LayoutTensor[mut=False, dtype, layout](a.unsafe_ptr())

        if use_simple:
            out_tensor = LayoutTensor[mut=False, dtype, layout](
                out.unsafe_ptr()
            )

            ctx.enqueue_function[prefix_sum_simple[layout]](
                out_tensor,
                a_tensor,
                size,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )
        else:
            var out_tensor = LayoutTensor[mut=False, dtype, extended_layout](
                out.unsafe_ptr()
            )

            # ANCHOR: prefix_sum_complete_block_level_sync
            # Phase 1: Local prefix sums
            ctx.enqueue_function[
                prefix_sum_local_phase[extended_layout, extended_layout]
            ](
                out_tensor,
                a_tensor,
                size,
                num_blocks,
                grid_dim=BLOCKS_PER_GRID_2,
                block_dim=THREADS_PER_BLOCK_2,
            )

            # Wait for all `blocks` to complete with using host `ctx.synchronize()`
            # Note this is in contrast with using `barrier()` in the kernel
            # which is a synchronization point for all threads in the same block and not across blocks.
            ctx.synchronize()

            # Phase 2: Add block sums
            ctx.enqueue_function[prefix_sum_block_sum_phase[extended_layout]](
                out_tensor,
                size,
                grid_dim=BLOCKS_PER_GRID_2,
                block_dim=THREADS_PER_BLOCK_2,
            )
            # ANCHOR_END: prefix_sum_complete_block_level_sync

        # Verify results for both cases
        expected = ctx.enqueue_create_host_buffer[dtype](size).enqueue_fill(0)
        ctx.synchronize()

        with a.map_to_host() as a_host:
            expected[0] = a_host[0]
            for i in range(1, size):
                expected[i] = expected[i - 1] + a_host[i]

        with out.map_to_host() as out_host:
            if not use_simple:
                print(
                    "Note: we print the extended buffer here, but we only need"
                    " to print the first `size` elements"
                )

            print("out:", out_host)
            print("expected:", expected)
            # Here we need to use the size of the original array, not the extended one
            size = size if use_simple else SIZE_2
            for i in range(size):
                assert_equal(out_host[i], expected[i])
