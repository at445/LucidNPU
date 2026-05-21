/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Declarations                                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: Ops.td                                                               *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace lucid_frontend {
class ConstantOp;
} // namespace lucid_frontend
#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

namespace lucid_frontend {

//===----------------------------------------------------------------------===//
// ::lucid_frontend::ConstantOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ConstantOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;

public:
  ConstantOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {},
                               const ::mlir::EmptyProperties &properties = {},
                               ::mlir::RegionRange regions = {})
      : odsAttrs(attrs), odsRegions(regions) {
    if (odsAttrs)
      odsOpName.emplace("toy.constant", odsAttrs.getContext());
  }

  ConstantOpGenericAdaptorBase(::mlir::Operation *op)
      : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()),
        odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned>
  getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() { return odsAttrs; }
};
} // namespace detail
template <typename RangeT>
class ConstantOpGenericAdaptor : public detail::ConstantOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ConstantOpGenericAdaptorBase;

public:
  ConstantOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {},
                           const ::mlir::EmptyProperties &properties = {},
                           ::mlir::RegionRange regions = {})
      : Base(attrs, properties, regions), odsOperands(values) {}

  ConstantOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs,
                           ::mlir::OpaqueProperties properties,
                           ::mlir::RegionRange regions = {})
      : ConstantOpGenericAdaptor(
            values, attrs,
            (properties ? *properties.as<::mlir::EmptyProperties *>()
                        : ::mlir::EmptyProperties{}),
            regions) {}

  template <typename LateInst = ConstantOp,
            typename = std::enable_if_t<std::is_same_v<LateInst, ConstantOp>>>
  ConstantOpGenericAdaptor(RangeT values, LateInst op)
      : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {
        std::next(odsOperands.begin(), valueRange.first),
        std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() { return odsOperands; }

private:
  RangeT odsOperands;
};
class ConstantOpAdaptor : public ConstantOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ConstantOpGenericAdaptor::ConstantOpGenericAdaptor;
  ConstantOpAdaptor(ConstantOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ConstantOp
    : public ::mlir::Op<
          ConstantOp, ::mlir::OpTrait::ZeroRegions,
          ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors,
          ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConstantOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ConstantOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() { return {}; }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("toy.constant");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
            std::next(getOperation()->operand_begin(),
                      valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
            std::next(getOperation()->result_begin(),
                      valueRange.first + valueRange.second)};
  }

  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState);
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState,
                    ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState,
                    ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands,
                    ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();

public:
};
} // namespace lucid_frontend
MLIR_DECLARE_EXPLICIT_TYPE_ID(::lucid_frontend::ConstantOp)

#endif // GET_OP_CLASSES
