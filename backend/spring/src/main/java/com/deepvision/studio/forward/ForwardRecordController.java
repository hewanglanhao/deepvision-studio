package com.deepvision.studio.forward;

import com.deepvision.studio.auth.AppUser;
import com.deepvision.studio.auth.AppUserRepository;
import com.deepvision.studio.forward.ForwardRecordDtos.ForwardRecordDetail;
import com.deepvision.studio.forward.ForwardRecordDtos.ForwardRecordSummary;
import com.deepvision.studio.forward.ForwardRecordDtos.SaveForwardRecordRequest;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.security.SecurityRequirement;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import java.security.Principal;
import java.util.List;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/a/forward-records")
@Tag(name = "Mode A Records", description = "Saved forward-pass snapshots for authenticated users")
@SecurityRequirement(name = "bearerAuth")
public class ForwardRecordController {
  private final ForwardRecordRepository records;
  private final AppUserRepository users;
  private final LocalImageStorage imageStorage;
  private final ObjectMapper objectMapper;

  /** 注入记录仓库、用户仓库、图片存储和 JSON 工具；保存记录时会同时落盘预览图和完整页面快照。 */
  public ForwardRecordController(
      ForwardRecordRepository records,
      AppUserRepository users,
      LocalImageStorage imageStorage,
      ObjectMapper objectMapper
  ) {
    this.records = records;
    this.users = users;
    this.imageStorage = imageStorage;
    this.objectMapper = objectMapper;
  }

  @GetMapping
  @Operation(summary = "List Mode A saved records for the current user")
  @ApiResponse(responseCode = "200", description = "Saved record summaries")
  @ApiResponse(responseCode = "401", description = "JWT is missing or invalid")
  /** 查询当前登录用户的 A 模式记录摘要，列表只展示元数据，不返回完整 snapshot。 */
  public List<ForwardRecordSummary> list(Principal principal) {
    return records.findByUserUsernameOrderByCreatedAtDesc(principal.getName()).stream()
        .map(ForwardRecordSummary::from)
        .toList();
  }

  @PostMapping
  @Operation(summary = "Save a Mode A forward-pass snapshot")
  @ApiResponse(responseCode = "200", description = "Saved record detail")
  @ApiResponse(responseCode = "400", description = "Invalid snapshot or preview image")
  @ApiResponse(responseCode = "401", description = "JWT is missing or invalid")
  /** 保存 A 模式实验快照：预览图落盘，网络结构和 forwardResult 序列化到 snapshotJson。 */
  public ForwardRecordDetail create(
      Principal principal,
      @Valid @RequestBody SaveForwardRecordRequest request
  ) throws JsonProcessingException {
    AppUser user = users.findByUsername(principal.getName())
        .orElseThrow(() -> new IllegalArgumentException("User not found."));
    String imagePath = imageStorage.saveDataUrl(user.getId(), request.previewImageDataUrl());
    String snapshotJson = objectMapper.writeValueAsString(request.snapshot());
    ForwardRecord record = records.save(new ForwardRecord(
        user,
        request.name().trim(),
        request.templateId(),
        request.datasetName(),
        request.layerCount(),
        request.parameterCount(),
        imagePath,
        snapshotJson
    ));
    return ForwardRecordDetail.from(record, objectMapper.readTree(record.getSnapshotJson()));
  }

  @GetMapping("/{id}")
  @Operation(summary = "Get a saved Mode A record detail")
  @ApiResponse(responseCode = "200", description = "Saved record detail")
  @ApiResponse(responseCode = "400", description = "Record not found for current user")
  @ApiResponse(responseCode = "401", description = "JWT is missing or invalid")
  /** 读取当前用户的一条记录详情，并把 snapshotJson 还原成前端可恢复页面状态的 JSON。 */
  public ForwardRecordDetail detail(Principal principal, @PathVariable Long id) throws JsonProcessingException {
    ForwardRecord record = records.findByIdAndUserUsername(id, principal.getName())
        .orElseThrow(() -> new IllegalArgumentException("Record not found."));
    JsonNode snapshot = objectMapper.readTree(record.getSnapshotJson());
    return ForwardRecordDetail.from(record, snapshot);
  }

  @DeleteMapping("/{id}")
  @Operation(summary = "Delete a saved Mode A record")
  @ApiResponse(responseCode = "200", description = "Record deleted")
  @ApiResponse(responseCode = "400", description = "Record not found for current user")
  @ApiResponse(responseCode = "401", description = "JWT is missing or invalid")
  /** 删除当前用户的一条记录；查询条件带用户名，避免用户访问或删除别人的快照。 */
  public void delete(Principal principal, @PathVariable Long id) {
    ForwardRecord record = records.findByIdAndUserUsername(id, principal.getName())
        .orElseThrow(() -> new IllegalArgumentException("Record not found."));
    records.delete(record);
  }
}
