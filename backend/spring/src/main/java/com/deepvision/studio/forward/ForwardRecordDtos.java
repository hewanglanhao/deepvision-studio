package com.deepvision.studio.forward;

import com.fasterxml.jackson.databind.JsonNode;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import java.time.Instant;

public final class ForwardRecordDtos {
  private ForwardRecordDtos() {}

  public record SaveForwardRecordRequest(
      @NotBlank(message = "Record name is required.")
      @Size(max = 120, message = "Record name must be at most 120 characters.")
      String name,
      @NotBlank(message = "Template id is required.")
      String templateId,
      @NotBlank(message = "Dataset name is required.")
      String datasetName,
      int layerCount,
      long parameterCount,
      String previewImageDataUrl,
      @NotNull(message = "Snapshot is required.")
      JsonNode snapshot
  ) {}

  public record ForwardRecordSummary(
      Long id,
      String name,
      String templateId,
      String datasetName,
      int layerCount,
      long parameterCount,
      String imagePath,
      Instant createdAt
  ) {
    static ForwardRecordSummary from(ForwardRecord record) {
      return new ForwardRecordSummary(
          record.getId(),
          record.getName(),
          record.getTemplateId(),
          record.getDatasetName(),
          record.getLayerCount(),
          record.getParameterCount(),
          record.getImagePath(),
          record.getCreatedAt()
      );
    }
  }

  public record ForwardRecordDetail(
      Long id,
      String name,
      String templateId,
      String datasetName,
      int layerCount,
      long parameterCount,
      String imagePath,
      Instant createdAt,
      JsonNode snapshot
  ) {
    static ForwardRecordDetail from(ForwardRecord record, JsonNode snapshot) {
      return new ForwardRecordDetail(
          record.getId(),
          record.getName(),
          record.getTemplateId(),
          record.getDatasetName(),
          record.getLayerCount(),
          record.getParameterCount(),
          record.getImagePath(),
          record.getCreatedAt(),
          snapshot
      );
    }
  }
}

