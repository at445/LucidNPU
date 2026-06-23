//===- Ops.td - Toy dialect operation definitions ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the operations of the Toy dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TOY_OPS
#define TOY_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td" 
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Provide a definition of the 'toy' dialect in the ODS framework so that we
// can define our operations.
def Toy_Dialect : Dialect {
  let name = "toy";
  let cppNamespace = "::mlir::toy";
}

// Base class for toy dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Toy_Op<string mnemonic, list<Trait> traits = []> :
    Op<Toy_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Toy Operations
//===----------------------------------------------------------------------===//

// marked as 'Pure' as it is a pure operation and may be removed if dead.
def ConstantOp : Toy_Op<"constant", [Pure]> {

    let summary = "constant";

    let arguments = (ins F64ElementsAttr:$value);

    let results = (outs F64Tensor);

    let hasCustomAssemblyFormat = 1;

    let builders = [
        // Build a constant with a given constant tensor value.
        OpBuilder<(ins "DenseElementsAttr":$value), [{
            build($_builder, $_state, value.getType(), value);
        }]>,

        // Build a constant with a given constant floating-point value.
        OpBuilder<(ins "double":$value)>
    ];

    let hasVerifier = 1;
}

def AddOp : Toy_Op<"add", [Pure, SameOperandsAndResultType]> {
    let summary = "element-wise addition operation";

    let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);

    let results = (outs F64Tensor:$result);

    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def SubOp : Toy_Op<"sub", [Pure, SameOperandsAndResultType]> {
    let summary = "element-wise subtraction operation";

    let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);

    let results = (outs F64Tensor:$result);

    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def MulOp : Toy_Op<"mul", [Pure, SameOperandsAndResultType]> {
    let summary = "element-wise multiplication operation";

    let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);

    let results = (outs F64Tensor:$result);

    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def DivOp : Toy_Op<"div", [Pure, SameOperandsAndResultType]> {
    let summary = "element-wise division operation";

    let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);

    let results = (outs F64Tensor:$result);

    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def MatrixMulOp : Toy_Op<"matrix_mul", [Pure]> {
    let summary = "matrix multiplication operation";

    let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);

    let results = (outs F64Tensor:$result);

    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";

    let builders = [
        OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
    ];

    // Indicate that additional verification for this operation is necessary.
    let hasVerifier = 1;
}

def ReturnOp : Toy_Op<"return", [HasParent<"FuncOp">, Terminator]> {
    let summary = "return operation";

    let arguments = (ins Variadic<F64Tensor>:$input);
    let assemblyFormat = "($input^ `:` type($input))? attr-dict";
}

def FuncOp : Toy_Op<"func", [
    FunctionOpInterface, IsolatedFromAbove
  ]> {
    let summary = "user defined function operation";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs
    );

    let regions = (region AnyRegion:$body);

    let builders = [OpBuilder<(ins
        "StringRef":$name, "FunctionType":$type,
        CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
    >];

    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // FunctionOpInterface Methods
        //===------------------------------------------------------------------===//

        /// Returns the argument types of this function.
        ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

        /// Returns the result types of this function.
        ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

        Region *getCallableRegion() { return &getBody(); }
    }];

    let hasCustomAssemblyFormat = 1;
    let skipDefaultBuilders = 1;
}
#endif // TOY_OPS
