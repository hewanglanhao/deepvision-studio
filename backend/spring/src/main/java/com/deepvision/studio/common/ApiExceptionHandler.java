package com.deepvision.studio.common;

import java.util.Map;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.AccessDeniedException;
import org.springframework.validation.FieldError;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.multipart.MaxUploadSizeExceededException;

@RestControllerAdvice
public class ApiExceptionHandler {
  @ExceptionHandler(IllegalArgumentException.class)
  /** 把业务校验失败转换成 400，前端可直接把错误展示在登录、上传或推理表单上。 */
  ResponseEntity<Map<String, String>> badRequest(IllegalArgumentException ex) {
    return ResponseEntity.badRequest().body(Map.of("error", ex.getMessage()));
  }

  @ExceptionHandler(AccessDeniedException.class)
  /** 把权限不足转换成 403，防止用户访问不属于自己的实验记录。 */
  ResponseEntity<Map<String, String>> accessDenied(AccessDeniedException ex) {
    return ResponseEntity.status(HttpStatus.FORBIDDEN).body(Map.of("error", ex.getMessage()));
  }

  @ExceptionHandler(IllegalStateException.class)
  /** 把后端依赖不可用转换成 503，例如 Python forward 服务或 LLM 代理暂时不可达。 */
  ResponseEntity<Map<String, String>> unavailable(IllegalStateException ex) {
    return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE).body(Map.of("error", ex.getMessage()));
  }

  @ExceptionHandler(MethodArgumentNotValidException.class)
  /** 提取 Bean Validation 的首个字段错误，让前端表单给出明确的用户名、密码或上传参数提示。 */
  ResponseEntity<Map<String, String>> validation(MethodArgumentNotValidException ex) {
    String message = ex.getBindingResult().getFieldErrors().stream()
        .findFirst()
        .map(FieldError::getDefaultMessage)
        .orElse("Invalid request.");
    return ResponseEntity.badRequest().body(Map.of("error", message));
  }

  @ExceptionHandler(MaxUploadSizeExceededException.class)
  /** 统一处理数据集上传过大，避免训练或推理样本把后端内存和磁盘写满。 */
  ResponseEntity<Map<String, String>> uploadTooLarge(MaxUploadSizeExceededException ex) {
    return ResponseEntity.status(HttpStatus.PAYLOAD_TOO_LARGE)
        .body(Map.of("error", "Uploaded dataset is too large. Please keep one file under 200MB and the whole request under 220MB."));
  }
}
