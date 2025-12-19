import os

import onnx
import torch
from onnx.external_data_helper import load_external_data_for_model
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process


def convert_final_fixed():
    output_file = "cosplace_resnet50.onnx"
    preprocessed_file = "cosplace_resnet50_pre.onnx"
    quantized_file = "cosplace_resnet50_int8.onnx"

    # 1. PyTorch 모델 로드
    print("1. 모델 로드 중...")
    try:
        model = torch.hub.load("gmberton/CosPlace", "get_trained_model", backbone="ResNet50", fc_output_dim=512)
        model.eval().cpu()
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return

    # 2. ONNX Export (Opset 18)
    print("2. ONNX Export (Opset 18)...")
    dummy_input = torch.randn(1, 3, 480, 640)
    torch.onnx.export(
        model,
        dummy_input,
        output_file,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"-> Export 완료: {output_file}")

    # [핵심 수정] 2.5 외부 데이터 합치기 (Monolithic 변환)
    # .data 파일이 따로 생겼을 경우를 대비해, 메모리로 로드해서 파일 하나로 다시 저장합니다.
    print("2.5. 파일 하나로 합치기 (Merging external data)...")
    try:
        onnx_model = onnx.load(output_file)
        # 외부 데이터가 있다면 메모리로 로드
        load_external_data_for_model(onnx_model, ".")
        # 다시 저장 (ResNet50은 2GB보다 작으므로 하나로 합쳐집니다)
        onnx.save(onnx_model, output_file)
        print("-> 병합 완료. 이제 안전하게 전처리할 수 있습니다.")
    except Exception as e:
        print(f"병합 과정 경고 (무시 가능): {e}")

    # 3. Pre-processing
    print("3. Pre-processing (Optimizing)...")
    try:
        quant_pre_process(
            input_model_path=output_file,
            output_model_path=preprocessed_file,
            skip_symbolic_shape=True,  # 이전에 발생한 NoneType 에러 방지
        )
        print(f"-> 전처리 완료: {preprocessed_file}")
    except Exception as e:
        print(f"전처리 실패: {e}")
        print("-> 원본 파일로 계속 진행합니다.")
        preprocessed_file = output_file

    # 3.5. Shape Info 제거 (충돌 방지)
    # quant_pre_process 후 Shape 정보가 꼬여서 quantize_dynamic에서 에러가 나는 경우가 많음
    # 따라서 명시적인 Shape 정보를 날리고 다시 추론하게 함
    print("3.5. Shape Info 제거 (충돌 방지)...")
    try:
        m = onnx.load(preprocessed_file)
        if len(m.graph.value_info) > 0:
            print(f"-> 기존 value_info {len(m.graph.value_info)}개 제거")
            del m.graph.value_info[:]
            onnx.save(m, preprocessed_file)
    except Exception as e:
        print(f"Shape 제거 중 오류 (무시): {e}")

    # 4. INT8 양자화
    print("4. INT8 양자화...")
    try:
        quantize_dynamic(model_input=preprocessed_file, model_output=quantized_file, weight_type=QuantType.QUInt8)
        print(f"\n🎉 성공! 최종 파일: {quantized_file}")
    except Exception as e:
        print(f"양자화 실패: {e}")


if __name__ == "__main__":
    convert_final_fixed()
