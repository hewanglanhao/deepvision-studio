package com.deepvision.studio.quiz;

import com.deepvision.studio.auth.AppUser;
import com.deepvision.studio.auth.AppUserRepository;
import com.deepvision.studio.quiz.QuizDtos.QuizAnswerRequest;
import com.deepvision.studio.quiz.QuizDtos.QuizAnswerResponse;
import com.deepvision.studio.quiz.QuizDtos.QuizAttemptSummary;
import com.deepvision.studio.quiz.QuizDtos.QuizProfileResponse;
import com.deepvision.studio.quiz.QuizDtos.QuizRecommendationResponse;
import com.deepvision.studio.quiz.QuizDtos.QuizQuestionResponse;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.annotation.PostConstruct;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class QuizService {
  private static final TypeReference<List<String>> STRING_LIST = new TypeReference<>() {};

  private final QuizQuestionRepository questions;
  private final QuizUserProfileRepository profiles;
  private final QuizAttemptRepository attempts;
  private final AppUserRepository users;
  private final ObjectMapper objectMapper;

  public QuizService(
      QuizQuestionRepository questions,
      QuizUserProfileRepository profiles,
      QuizAttemptRepository attempts,
      AppUserRepository users,
      ObjectMapper objectMapper
  ) {
    this.questions = questions;
    this.profiles = profiles;
    this.attempts = attempts;
    this.users = users;
    this.objectMapper = objectMapper;
  }

  @PostConstruct
  @Transactional
  public void seedQuestions() {
    for (QuestionSeed seed : questionSeeds()) {
      QuizQuestion next = new QuizQuestion(
          seed.code(),
          seed.topic(),
          seed.difficulty(),
          seed.prompt(),
          writeOptions(seed.options()),
          seed.answerIndex(),
          seed.explanation()
      );
      questions.findById(seed.code()).ifPresentOrElse(existing -> existing.replaceWith(next), () -> questions.save(next));
    }
  }

  @Transactional
  public QuizProfileResponse profile(String username) {
    return toProfileResponse(profileFor(user(username)));
  }

  @Transactional(readOnly = true)
  public List<QuizQuestionResponse> allQuestions() {
    return questions.findByActiveTrueOrderByTopicAscDifficultyAscCodeAsc().stream()
        .map(question -> toQuestionResponse(question, "题库练习"))
        .toList();
  }

  @Transactional
  public QuizRecommendationResponse recommended(String username, String mode, int limit) {
    AppUser user = user(username);
    QuizUserProfile profile = profileFor(user);
    List<QuizQuestion> all = new ArrayList<>(questions.findByActiveTrueOrderByTopicAscDifficultyAscCodeAsc());
    List<QuizAttempt> recentAttempts = attempts.findTop200ByUserOrderByAnsweredAtDesc(user);
    String normalizedMode = normalizeMode(mode);
    int cappedLimit = Math.max(1, Math.min(30, limit));
    List<QuizQuestionResponse> result = switch (normalizedMode) {
      case "spaced" -> spacedReviewQuestions(user, profile, all, recentAttempts, cappedLimit);
      case "exam" -> examQuestions(profile, all, cappedLimit);
      default -> weaknessQuestions(user, profile, all, cappedLimit);
    };
    return new QuizRecommendationResponse(
        normalizedMode,
        strategyName(normalizedMode),
        strategyDescription(normalizedMode),
        result
    );
  }

  @Transactional
  public QuizAnswerResponse answer(String username, QuizAnswerRequest request) {
    AppUser user = user(username);
    QuizQuestion question = questions.findById(request.questionCode())
        .filter(QuizQuestion::isActive)
        .orElseThrow(() -> new IllegalArgumentException("Question is not available."));
    List<String> options = readOptions(question);
    if (request.selectedIndex() < 0 || request.selectedIndex() >= options.size()) {
      throw new IllegalArgumentException("Selected option index is invalid.");
    }
    boolean correct = request.selectedIndex() == question.getAnswerIndex();
    QuizUserProfile profile = profileFor(user);
    profile.applyAnswer(question.getTopic(), correct, question.getDifficulty());
    attempts.save(new QuizAttempt(user, question, request.selectedIndex(), correct));
    return new QuizAnswerResponse(
        question.getCode(),
        correct,
        request.selectedIndex(),
        question.getAnswerIndex(),
        question.getExplanation(),
        toProfileResponse(profile)
    );
  }

  private AppUser user(String username) {
    if (username == null || username.isBlank()) {
      throw new IllegalArgumentException("Please login first.");
    }
    return users.findByUsername(username)
        .orElseThrow(() -> new IllegalArgumentException("User not found."));
  }

  private QuizUserProfile profileFor(AppUser user) {
    return profiles.findByUser(user).orElseGet(() -> profiles.save(new QuizUserProfile(user)));
  }

  private QuizProfileResponse toProfileResponse(QuizUserProfile profile) {
    int answered = profile.getAnsweredCount();
    double accuracy = answered <= 0 ? 0 : (double) profile.getCorrectCount() / answered;
    return new QuizProfileResponse(
        profile.scores(),
        answered,
        profile.getCorrectCount(),
        accuracy,
        profile.getCurrentStreak(),
        profile.getUpdatedAt(),
        attempts.findTop20ByUserOrderByAnsweredAtDesc(profile.getUser()).stream()
            .map(this::toAttemptSummary)
            .toList()
    );
  }

  private QuizQuestionResponse toQuestionResponse(QuizQuestion question, String reason) {
    return new QuizQuestionResponse(
        question.getCode(),
        question.getTopic(),
        question.getDifficulty(),
        question.getPrompt(),
        readOptions(question),
        reason
    );
  }

  private QuizAttemptSummary toAttemptSummary(QuizAttempt attempt) {
    QuizQuestion question = attempt.getQuestion();
    return new QuizAttemptSummary(
        question.getCode(),
        question.getTopic(),
        question.getDifficulty(),
        attempt.isCorrect(),
        attempt.getAnsweredAt()
    );
  }

  private List<QuizQuestionResponse> weaknessQuestions(
      AppUser user,
      QuizUserProfile profile,
      List<QuizQuestion> all,
      int limit
  ) {
    all.sort(Comparator
        .comparingDouble((QuizQuestion question) -> weaknessScore(user, profile, question))
        .thenComparing(QuizQuestion::getCode));
    return all.stream()
        .limit(limit)
        .map(question -> toQuestionResponse(question, weaknessReason(profile, question)))
        .toList();
  }

  private double weaknessScore(AppUser user, QuizUserProfile profile, QuizQuestion question) {
    double mastery = profile.score(question.getTopic());
    double targetDifficulty = targetDifficulty(mastery);
    double difficultyGap = Math.abs(question.getDifficulty() - targetDifficulty);
    double repetitionPenalty = attempts.existsByUserAndQuestion(user, question) ? 0.8 : 0;
    return mastery + difficultyGap * 8 + repetitionPenalty;
  }

  private String weaknessReason(QuizUserProfile profile, QuizQuestion question) {
    double mastery = profile.score(question.getTopic());
    return "薄弱维度 " + topicLabel(question.getTopic()) + " 当前约 " + Math.round(mastery)
        + " 分，题目难度贴近最近发展区。";
  }

  private List<QuizQuestionResponse> spacedReviewQuestions(
      AppUser user,
      QuizUserProfile profile,
      List<QuizQuestion> all,
      List<QuizAttempt> recentAttempts,
      int limit
  ) {
    Map<String, Instant> lastByTopic = new HashMap<>();
    Map<String, Integer> wrongByTopic = new HashMap<>();
    Map<String, Instant> lastByQuestion = new HashMap<>();
    for (QuizAttempt attempt : recentAttempts) {
      QuizQuestion question = attempt.getQuestion();
      lastByTopic.putIfAbsent(question.getTopic(), attempt.getAnsweredAt());
      lastByQuestion.putIfAbsent(question.getCode(), attempt.getAnsweredAt());
      if (!attempt.isCorrect()) {
        wrongByTopic.merge(question.getTopic(), 1, Integer::sum);
      }
    }
    Instant now = Instant.now();
    all.sort(Comparator
        .comparingDouble((QuizQuestion question) -> -spacedPriority(profile, question, lastByTopic, lastByQuestion, wrongByTopic, now))
        .thenComparing(QuizQuestion::getCode));
    return all.stream()
        .limit(limit)
        .map(question -> toQuestionResponse(question, spacedReason(question, lastByTopic, wrongByTopic, now)))
        .toList();
  }

  private double spacedPriority(
      QuizUserProfile profile,
      QuizQuestion question,
      Map<String, Instant> lastByTopic,
      Map<String, Instant> lastByQuestion,
      Map<String, Integer> wrongByTopic,
      Instant now
  ) {
    long topicHours = hoursSince(lastByTopic.get(question.getTopic()), now);
    long questionHours = hoursSince(lastByQuestion.get(question.getCode()), now);
    double masteryGap = Math.max(0, 80 - profile.score(question.getTopic()));
    double wrongBoost = wrongByTopic.getOrDefault(question.getTopic(), 0) * 8.0;
    double due = Math.min(72, topicHours) + Math.min(120, questionHours) * 0.25;
    return due + masteryGap * 0.8 + wrongBoost - question.getDifficulty() * 1.5;
  }

  private String spacedReason(QuizQuestion question, Map<String, Instant> lastByTopic, Map<String, Integer> wrongByTopic, Instant now) {
    long hours = hoursSince(lastByTopic.get(question.getTopic()), now);
    int wrong = wrongByTopic.getOrDefault(question.getTopic(), 0);
    if (hours >= 24) {
      return topicLabel(question.getTopic()) + " 已间隔 " + (hours / 24) + " 天未巩固，适合复习。";
    }
    if (wrong > 0) {
      return topicLabel(question.getTopic()) + " 最近有 " + wrong + " 次错误，优先安排巩固。";
    }
    return "用于保持 " + topicLabel(question.getTopic()) + " 的记忆稳定性。";
  }

  private List<QuizQuestionResponse> examQuestions(QuizUserProfile profile, List<QuizQuestion> all, int limit) {
    Map<String, List<QuizQuestion>> byTopic = new LinkedHashMap<>();
    for (QuizQuestion question : all) {
      byTopic.computeIfAbsent(question.getTopic(), ignored -> new ArrayList<>()).add(question);
    }
    byTopic.values().forEach(list -> list.sort(Comparator
        .comparingInt((QuizQuestion question) -> difficultyBucketRank(question.getDifficulty()))
        .thenComparingDouble(question -> profile.score(question.getTopic()))
        .thenComparing(QuizQuestion::getCode)));

    List<QuizQuestion> selected = new ArrayList<>();
    List<String> topics = new ArrayList<>(byTopic.keySet());
    int cursor = 0;
    while (selected.size() < limit && !topics.isEmpty()) {
      String topic = topics.get(cursor % topics.size());
      List<QuizQuestion> bucket = byTopic.get(topic);
      QuizQuestion next = bucket.stream()
          .filter(question -> !selected.contains(question))
          .findFirst()
          .orElse(null);
      if (next != null) {
        selected.add(next);
      }
      cursor += 1;
      if (cursor > topics.size() * 10 && selected.size() < limit) {
        break;
      }
    }
    if (selected.size() < limit) {
      for (QuizQuestion question : all) {
        if (selected.size() >= limit) break;
        if (!selected.contains(question)) {
          selected.add(question);
        }
      }
    }
    return selected.stream()
        .map(question -> toQuestionResponse(question, "套题覆盖 " + topicLabel(question.getTopic()) + "，难度 " + question.getDifficulty() + "。"))
        .toList();
  }

  private int difficultyBucketRank(int difficulty) {
    return switch (difficulty) {
      case 1 -> 0;
      case 2 -> 1;
      case 3 -> 2;
      default -> 3;
    };
  }

  private double targetDifficulty(double mastery) {
    if (mastery < 35) return 1.0;
    if (mastery < 55) return 2.0;
    if (mastery < 75) return 3.0;
    return 4.0;
  }

  private long hoursSince(Instant instant, Instant now) {
    if (instant == null) return 168;
    return Math.max(0, Duration.between(instant, now).toHours());
  }

  private String normalizeMode(String mode) {
    return switch (mode == null ? "" : mode.trim().toLowerCase()) {
      case "spaced", "spaced_review" -> "spaced";
      case "exam", "paper" -> "exam";
      default -> "weakness";
    };
  }

  private String strategyName(String mode) {
    return switch (mode) {
      case "spaced" -> "间隔复习策略";
      case "exam" -> "套题组卷模式";
      default -> "优先补弱策略";
    };
  }

  private String strategyDescription(String mode) {
    return switch (mode) {
      case "spaced" -> "结合最近练习时间、错题情况和掌握分变化，安排周期性巩固。";
      case "exam" -> "按知识点覆盖和难度比例组卷，形成一套更完整的测验。";
      default -> "围绕薄弱知识点优先推荐练习，并结合最近发展区控制难度。";
    };
  }

  private String topicLabel(String topic) {
    return switch (topic) {
      case "ai_foundations" -> "AI 基础";
      case "machine_learning" -> "机器学习";
      case "neural_networks" -> "神经网络";
      case "deep_learning_training" -> "深度学习训练";
      case "convolution_vision" -> "卷积与视觉";
      case "sequence_models" -> "序列模型";
      case "evaluation_metrics" -> "评估指标";
      case "responsible_ai" -> "负责任 AI";
      default -> topic;
    };
  }

  private String writeOptions(List<String> options) {
    try {
      return objectMapper.writeValueAsString(options);
    } catch (JsonProcessingException ex) {
      throw new IllegalArgumentException("Failed to serialize quiz options.");
    }
  }

  private List<String> readOptions(QuizQuestion question) {
    try {
      return objectMapper.readValue(question.getOptionsJson(), STRING_LIST);
    } catch (JsonProcessingException ex) {
      throw new IllegalArgumentException("Failed to parse quiz options.");
    }
  }

  private List<QuestionSeed> questionSeeds() {
    return List.of(
        q("g-ai-001", "ai_foundations", 1, "人工智能系统中的“智能体”通常指什么？", List.of("只能执行数学运算的函数", "能够感知环境并采取行动的系统", "只包含数据库的后端服务", "固定不变的静态网页"), 1, "智能体强调感知、决策和行动闭环。"),
        q("g-ai-002", "ai_foundations", 1, "监督学习和无监督学习的主要区别是什么？", List.of("是否使用 GPU", "是否有人工提供的标签", "模型是否有隐藏层", "数据是否能被压缩"), 1, "监督学习依赖输入与标签的对应关系，无监督学习主要发现数据结构。"),
        q("g-ai-003", "ai_foundations", 2, "以下哪项更符合“训练集”的作用？", List.of("用于最终上线后保存日志", "用于模型学习参数", "只用于展示图片预览", "只用于人工修改代码"), 1, "训练集参与参数更新，是模型学习规律的主要数据来源。"),
        q("g-ai-004", "ai_foundations", 2, "模型泛化能力指的是模型能够怎样表现？", List.of("只记住训练样本", "在未见过的数据上仍保持较好效果", "让参数数量无限增加", "避免任何数学计算"), 1, "泛化能力关注模型对新样本的表现。"),
        q("g-ai-005", "ai_foundations", 2, "人工智能中的“特征”通常指什么？", List.of("样本中可供模型使用的信息表示", "数据库的用户名", "网页按钮的颜色", "操作系统进程号"), 0, "特征是输入样本中用于预测或判断的变量。"),
        q("g-ai-006", "ai_foundations", 3, "如果一个模型在训练集效果很好但测试集效果很差，最可能的问题是？", List.of("欠拟合", "过拟合", "学习率恒为 0", "没有输入数据"), 1, "训练好、测试差是典型过拟合现象。"),

        q("g-ml-001", "machine_learning", 1, "分类任务的目标通常是什么？", List.of("预测连续数值", "把样本分到离散类别", "删除异常文件", "加密用户密码"), 1, "分类任务输出离散类别，如猫/狗或 0-9 数字。"),
        q("g-ml-002", "machine_learning", 1, "回归任务的输出通常是什么？", List.of("连续数值", "固定的四个类别", "图片文件名", "网络端口号"), 0, "回归用于预测价格、温度等连续值。"),
        q("g-ml-003", "machine_learning", 2, "学习率过大时，训练中常见现象是？", List.of("损失稳定下降到 0", "损失震荡甚至发散", "所有样本自动变多", "模型参数被冻结"), 1, "学习率过大会导致更新步子太大，损失可能震荡或发散。"),
        q("g-ml-004", "machine_learning", 2, "验证集的主要用途是？", List.of("参与每一步反向传播", "辅助选择超参数和观察泛化", "替代模型参数", "存放网页静态资源"), 1, "验证集通常不直接训练参数，而用于调参和早停判断。"),
        q("g-ml-005", "machine_learning", 3, "正则化的核心目的是什么？", List.of("降低模型对训练集细节的过度依赖", "删除输出层", "把标签改成随机值", "让 batch size 等于 1"), 0, "正则化通过约束复杂度等方式缓解过拟合。"),
        q("g-ml-006", "machine_learning", 3, "交叉熵损失常用于哪类任务？", List.of("图像压缩", "分类", "文件上传", "数据库备份"), 1, "交叉熵衡量预测类别分布与真实类别之间的差异。"),

        q("g-nn-001", "neural_networks", 1, "神经网络中的权重参数主要用于什么？", List.of("保存用户头像", "控制输入到输出的映射关系", "确定浏览器窗口大小", "记录文件路径"), 1, "权重决定神经元之间信号如何组合和传递。"),
        q("g-nn-002", "neural_networks", 1, "激活函数的作用通常是？", List.of("引入非线性表达能力", "删除所有负样本", "替代数据库主键", "停止模型训练"), 0, "没有非线性激活，多层线性变换仍等价于线性模型。"),
        q("g-nn-003", "neural_networks", 2, "ReLU(x) 的输出形式是？", List.of("恒等于 1", "max(0, x)", "x 的平方", "随机类别"), 1, "ReLU 将负值截断为 0，正值保持不变。"),
        q("g-nn-004", "neural_networks", 2, "全连接层的特点是？", List.of("每个输出单元通常连接全部输入单元", "只能处理图片", "没有可训练参数", "只能放在输入层之前"), 0, "全连接层通过权重矩阵把输入向量映射到输出向量。"),
        q("g-nn-005", "neural_networks", 3, "梯度消失会直接导致什么问题？", List.of("前面层参数更新很弱", "训练样本自动打乱失败", "图片无法显示", "标签列变为空"), 0, "梯度很小时，靠近输入端的层学习会变慢。"),
        q("g-nn-006", "neural_networks", 3, "残差连接的主要思想是？", List.of("完全移除卷积层", "让层学习相对输入的增量并保留 shortcut", "只训练输出偏置", "把学习率设为负数"), 1, "残差块通过 shortcut 缓解深层网络训练困难。"),

        q("g-train-001", "deep_learning_training", 1, "反向传播主要用于计算什么？", List.of("每个参数对损失的梯度", "图片文件大小", "用户登录时间", "数据集所在端口"), 0, "反向传播基于链式法则计算参数梯度。"),
        q("g-train-002", "deep_learning_training", 1, "优化器的作用是？", List.of("根据梯度更新参数", "显示网页标题", "校验压缩包格式", "生成用户名"), 0, "优化器如 SGD、Adam 会使用梯度决定参数更新。"),
        q("g-train-003", "deep_learning_training", 2, "Batch size 表示什么？", List.of("一次参数更新中使用的样本数量", "网络层数量", "类别总数的平方", "图片通道数"), 0, "Batch size 决定每次迭代使用多少样本估计梯度。"),
        q("g-train-004", "deep_learning_training", 2, "Adam 优化器相比普通 SGD，常见特点是？", List.of("使用一阶和二阶矩估计自适应调整步长", "不需要损失函数", "只能训练线性回归", "禁止使用 mini-batch"), 0, "Adam 会维护梯度均值和平方梯度均值。"),
        q("g-train-005", "deep_learning_training", 3, "梯度爆炸时，以下哪种方法常被使用？", List.of("梯度裁剪", "删除训练集", "把所有标签设为 0", "关闭损失函数"), 0, "梯度裁剪可以限制梯度范数，缓解不稳定更新。"),
        q("g-train-006", "deep_learning_training", 3, "学习率调度器的作用是？", List.of("在训练过程中调整学习率", "改变数据库表名", "把图片转成 zip", "删除验证集"), 0, "学习率调度器可按轮次、指标或策略改变学习率。"),
        q("g-train-007", "deep_learning_training", 4, "早停策略通常依据什么决定停止训练？", List.of("验证集指标长期不再改善", "训练日志字体大小", "用户头像颜色", "操作系统版本"), 0, "早停用验证集趋势避免继续过拟合或浪费训练时间。"),

        q("g-cv-001", "convolution_vision", 1, "卷积层最常用于处理哪类结构化数据？", List.of("具有空间局部结构的图像", "纯文本密码", "数据库连接池", "端口映射配置"), 0, "卷积利用局部感受野和权重共享，适合图像。"),
        q("g-cv-002", "convolution_vision", 1, "RGB 图片通常有几个通道？", List.of("1", "2", "3", "10"), 2, "RGB 分别对应红、绿、蓝三个通道。"),
        q("g-cv-003", "convolution_vision", 2, "池化层常见作用是？", List.of("降低空间尺寸并保留重要响应", "增加标签数量", "保存 checkpoint", "替代所有激活函数"), 0, "池化可降低特征图大小，提高一定平移鲁棒性。"),
        q("g-cv-004", "convolution_vision", 2, "1x1 卷积常用于什么？", List.of("调整通道数或做通道混合", "改变图片文件扩展名", "生成随机标签", "停止梯度计算"), 0, "1x1 卷积可在不直接扩大空间感受野的情况下改变通道维度。"),
        q("g-cv-005", "convolution_vision", 3, "卷积核数量通常决定输出特征图的什么？", List.of("通道数", "训练样本数", "类别名称", "学习率小数位"), 0, "每个卷积核通常产生一个输出通道。"),
        q("g-cv-006", "convolution_vision", 3, "在 CNN 中 Flatten 层的主要作用是？", List.of("把多维特征图展平成向量以接全连接层", "把标签变成图片", "删除 batch 维度", "自动下载数据集"), 0, "Flatten 常用于卷积特征到 MLP 分类头的过渡。"),

        q("g-seq-001", "sequence_models", 1, "序列模型更适合处理哪类数据？", List.of("有时间或顺序关系的数据", "完全无序的颜色表", "固定大小的配置文件名", "单个布尔值"), 0, "文本、语音、时间序列都包含顺序依赖。"),
        q("g-seq-002", "sequence_models", 2, "RNN 的核心特点是？", List.of("隐藏状态在时间步之间传递信息", "只能处理 32x32 图片", "没有参数", "只能输出二分类"), 0, "RNN 通过隐藏状态建模历史信息。"),
        q("g-seq-003", "sequence_models", 2, "LSTM/GRU 相比普通 RNN 主要改进了什么？", List.of("用门控机制缓解长依赖训练困难", "取消所有矩阵乘法", "只能运行在浏览器", "不再需要输入"), 0, "门控结构帮助保留或遗忘信息。"),
        q("g-seq-004", "sequence_models", 3, "Transformer 中注意力机制的核心作用是？", List.of("根据相关性聚合不同位置的信息", "把所有词随机打乱", "压缩数据库文件", "禁止并行计算"), 0, "注意力根据 query-key 相似度加权 value。"),
        q("g-seq-005", "sequence_models", 3, "位置编码在 Transformer 中的意义是？", List.of("提供 token 的顺序信息", "保存用户密码", "替代损失函数", "固定 batch size"), 0, "自注意力本身不含顺序，位置编码提供位置信息。"),
        q("g-seq-006", "sequence_models", 4, "多头注意力的直观好处是？", List.of("从多个表示子空间关注不同关系", "让模型没有参数", "只输出一个常数", "删除输入序列"), 0, "多头注意力允许模型并行学习不同类型的依赖。"),

        q("g-eval-001", "evaluation_metrics", 1, "分类准确率表示什么？", List.of("预测正确样本占总样本比例", "损失函数的梯度", "模型参数总字节数", "图片分辨率"), 0, "准确率是正确预测数除以样本总数。"),
        q("g-eval-002", "evaluation_metrics", 2, "混淆矩阵用于观察什么？", List.of("不同真实类别和预测类别的对应情况", "服务器内存占用", "图片压缩率", "代码行数"), 0, "混淆矩阵能展示哪些类别容易互相混淆。"),
        q("g-eval-003", "evaluation_metrics", 2, "Precision 更关注什么？", List.of("被预测为正类的样本中有多少是真的正类", "所有正类中被找回多少", "训练轮数", "输入图片宽度"), 0, "Precision 衡量正类预测的纯度。"),
        q("g-eval-004", "evaluation_metrics", 2, "Recall 更关注什么？", List.of("真实正类中有多少被模型找回", "模型参数是否保存", "网页是否加载", "学习率是否为整数"), 0, "Recall 衡量正类覆盖率。"),
        q("g-eval-005", "evaluation_metrics", 3, "F1 分数是什么的调和平均？", List.of("Precision 和 Recall", "Loss 和 Epoch", "Batch size 和通道数", "用户名和密码"), 0, "F1 兼顾 Precision 与 Recall。"),
        q("g-eval-006", "evaluation_metrics", 3, "回归任务中 MSE 衡量什么？", List.of("预测值与真实值误差平方的平均", "分类正确比例", "图片通道数量", "模型文件路径长度"), 0, "MSE 对较大误差惩罚更重。"),
        q("g-eval-007", "evaluation_metrics", 4, "测试集应尽量避免参与调参，主要原因是？", List.of("防止测试结果被间接过拟合", "测试集不能包含标签", "测试集只能是图片", "测试集会删除模型"), 0, "频繁用测试集调参会让评估不再客观。"),

        q("g-rai-001", "responsible_ai", 1, "数据隐私在 AI 项目中主要关注什么？", List.of("保护用户和样本中的敏感信息", "让模型层数增加", "让图片更清晰", "提高端口号"), 0, "上传数据和训练结果都可能包含敏感信息，需要限制访问。"),
        q("g-rai-002", "responsible_ai", 2, "数据偏差可能导致什么问题？", List.of("模型对某些群体或场景表现不公平", "模型一定变成无监督学习", "数据库无法启动", "所有预测都正确"), 0, "偏差会让模型在训练数据不足或偏斜的群体上表现较差。"),
        q("g-rai-003", "responsible_ai", 2, "可解释性工具的作用更接近于？", List.of("帮助理解模型为何做出某些预测", "替代所有训练数据", "自动保证模型公平", "删除错误样本"), 0, "可解释性帮助分析模型关注的特征、层激活或决策依据。"),
        q("g-rai-004", "responsible_ai", 3, "在教学平台中限制用户只能看自己的上传数据，主要属于哪类设计？", List.of("权限与隐私隔离", "卷积加速", "学习率调度", "数据增强"), 0, "用户私有数据应按账号隔离，避免互相泄露。"),
        q("g-rai-005", "responsible_ai", 3, "模型部署后仍需要监控的原因是？", List.of("真实数据分布可能随时间变化", "部署后模型会自动删除", "训练集会无限增大", "浏览器会替代后端"), 0, "数据漂移会导致线上效果下降，需要持续监控。"),
        q("g-rai-006", "responsible_ai", 4, "当 AI 系统用于高风险场景时，较合理的做法是？", List.of("加入人工审核和明确责任边界", "隐藏所有错误", "禁止记录日志", "只看训练集准确率"), 0, "高风险场景需要审计、人机协同和责任界定。")
    );
  }

  private QuestionSeed q(String code, String topic, int difficulty, String prompt, List<String> options, int answerIndex, String explanation) {
    return new QuestionSeed(code, topic, difficulty, prompt, options, answerIndex, explanation);
  }

  private record QuestionSeed(
      String code,
      String topic,
      int difficulty,
      String prompt,
      List<String> options,
      int answerIndex,
      String explanation
  ) {}
}
