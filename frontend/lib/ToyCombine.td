#ifndef TOY_COMBINE
#define TOY_COMBINE

include "mlir/IR/PatternBase.td"
include "include/Ops.td"

// transpose(transpose(x)) = x
def TransposeTransposeOptPattern : Pat<(TransposeOp(TransposeOp $arg)),
                                       (replaceWithValue $arg) >;

// reshape(reshape(x)) = reshape(x)
def ReshapeReshapeOptPattern : Pat< (ReshapeOp(ReshapeOp $arg)),
                                    (ReshapeOp $arg) >;


def TypesAreIdentical : Constraint<CPred<"$0.getType() == $1.getType()">>;
def RedundantReshapeOptPattern : Pat<
    (ReshapeOp:$res $arg), (replaceWithValue $arg),
    [(TypesAreIdentical $res, $arg)]
>;

def ReshapeConstant : NativeCodeCall<"$0.reshape(mlir::cast<ShapedType>(($1.getType())))">;
def FoldReshapeConstantOptPattern : Pat<
    (ReshapeOp:$res (ConstantOp $arg)),
    (ConstantOp (ReshapeConstant $arg, $res))
>;



#endif // TOY_COMBINE
