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
  struct Properties {
    using valueTy = ::mlir::DenseElementsAttr;
    valueTy value;

    auto getValue() {
      auto &propStorage = this->value;
      return ::llvm::cast<::mlir::DenseElementsAttr>(propStorage);
    }
    void setValue(const ::mlir::DenseElementsAttr &propValue) {
      this->value = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return rhs.value == this->value && true;
    }
    bool operator!=(const Properties &rhs) const { return !(*this == rhs); }
  };

protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;

public:
  ConstantOpGenericAdaptorBase(::mlir::DictionaryAttr attrs,
                               const Properties &properties,
                               ::mlir::RegionRange regions = {})
      : odsAttrs(attrs), properties(properties), odsRegions(regions) {
    if (odsAttrs)
      odsOpName.emplace("toy.constant", odsAttrs.getContext());
  }

  ConstantOpGenericAdaptorBase(ConstantOp op);

  std::pair<unsigned, unsigned>
  getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() { return properties; }

  ::mlir::DictionaryAttr getAttributes() { return odsAttrs; }

  ::mlir::DenseElementsAttr getValueAttr() {
    auto attr = ::llvm::cast<::mlir::DenseElementsAttr>(getProperties().value);
    return attr;
  }

  ::mlir::DenseElementsAttr getValue();
};
} // namespace detail
template <typename RangeT>
class ConstantOpGenericAdaptor : public detail::ConstantOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ConstantOpGenericAdaptorBase;

public:
  ConstantOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs,
                           const Properties &properties,
                           ::mlir::RegionRange regions = {})
      : Base(attrs, properties, regions), odsOperands(values) {}

  ConstantOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs,
                           ::mlir::OpaqueProperties properties,
                           ::mlir::RegionRange regions = {})
      : ConstantOpGenericAdaptor(
            values, attrs,
            (properties ? *properties.as<Properties *>() : Properties{}),
            regions) {}

  ConstantOpGenericAdaptor(RangeT values,
                           ::mlir::DictionaryAttr attrs = nullptr)
      : ConstantOpGenericAdaptor(values, attrs, Properties{}, {}) {}

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
          ConstantOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult,
          ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl,
          ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands,
          ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConstantOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ConstantOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("value")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getValueAttrName() { return getAttributeNameForIndex(0); }

  static ::mlir::StringAttr getValueAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

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

  static ::llvm::LogicalResult setPropertiesFromAttr(
      Properties &prop, ::mlir::Attribute attr,
      ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx,
                                               const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute>
  getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop,
                  llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name,
                              mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx,
                                    const Properties &prop,
                                    ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(
      ::mlir::OperationName opName, ::mlir::NamedAttrList &attrs,
      llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult
  readProperties(::mlir::DialectBytecodeReader &reader,
                 ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::DenseElementsAttr getValueAttr() {
    return ::llvm::cast<::mlir::DenseElementsAttr>(getProperties().value);
  }

  ::mlir::DenseElementsAttr getValue();
  void setValueAttr(::mlir::DenseElementsAttr attr) {
    getProperties().value = attr;
  }

  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState, ::mlir::Type resultType0,
                    ::mlir::DenseElementsAttr value);
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState,
                    ::mlir::TypeRange resultTypes,
                    ::mlir::DenseElementsAttr value);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState,
                    ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands,
                    ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();

private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name,
                                                     unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() &&
           "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
                                  "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace lucid_frontend
MLIR_DECLARE_EXPLICIT_TYPE_ID(::lucid_frontend::ConstantOp)

#endif // GET_OP_CLASSES
