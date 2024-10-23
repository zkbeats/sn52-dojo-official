-- CreateEnum
CREATE TYPE "CriteriaTypeEnum" AS ENUM ('RANKING_CRITERIA', 'MULTI_SCORE', 'SCORE', 'MULTI_SELECT');

-- CreateTable
CREATE TABLE "Ground_Truth_Model" (
    "id" TEXT NOT NULL,
    "request_id" TEXT NOT NULL,
    "obfuscated_model_id" TEXT NOT NULL,
    "real_model_id" TEXT NOT NULL,
    "rank_id" INTEGER NOT NULL,
    "feedback_request_id" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "Ground_Truth_Model_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Feedback_Request_Model" (
    "id" TEXT NOT NULL,
    "request_id" TEXT NOT NULL,
    "prompt" TEXT NOT NULL,
    "task_type" TEXT NOT NULL,
    "is_processed" BOOLEAN NOT NULL DEFAULT false,
    "dojo_task_id" TEXT,
    "hotkey" TEXT NOT NULL,
    "expire_at" TIMESTAMP(3) NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,
    "parent_id" TEXT,

    CONSTRAINT "Feedback_Request_Model_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Completion_Response_Model" (
    "id" TEXT NOT NULL,
    "completion_id" TEXT NOT NULL,
    "model" TEXT NOT NULL,
    "completion" JSONB NOT NULL,
    "rank_id" INTEGER,
    "score" DOUBLE PRECISION,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,
    "feedback_request_id" TEXT NOT NULL,

    CONSTRAINT "Completion_Response_Model_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Criteria_Type_Model" (
    "id" TEXT NOT NULL,
    "type" "CriteriaTypeEnum" NOT NULL,
    "options" JSONB NOT NULL,
    "min" DOUBLE PRECISION,
    "max" DOUBLE PRECISION,
    "feedback_request_id" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "Criteria_Type_Model_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Score_Model" (
    "id" TEXT NOT NULL,
    "score" JSONB NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "Score_Model_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "Ground_Truth_Model_request_id_obfuscated_model_id_rank_id_key" ON "Ground_Truth_Model"("request_id", "obfuscated_model_id", "rank_id");

-- CreateIndex
CREATE UNIQUE INDEX "Feedback_Request_Model_id_key" ON "Feedback_Request_Model"("id");

-- CreateIndex
CREATE UNIQUE INDEX "Feedback_Request_Model_request_id_hotkey_key" ON "Feedback_Request_Model"("request_id", "hotkey");

-- AddForeignKey
ALTER TABLE "Ground_Truth_Model" ADD CONSTRAINT "Ground_Truth_Model_feedback_request_id_fkey" FOREIGN KEY ("feedback_request_id") REFERENCES "Feedback_Request_Model"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Feedback_Request_Model" ADD CONSTRAINT "Feedback_Request_Model_parent_id_fkey" FOREIGN KEY ("parent_id") REFERENCES "Feedback_Request_Model"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Completion_Response_Model" ADD CONSTRAINT "Completion_Response_Model_feedback_request_id_fkey" FOREIGN KEY ("feedback_request_id") REFERENCES "Feedback_Request_Model"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Criteria_Type_Model" ADD CONSTRAINT "Criteria_Type_Model_feedback_request_id_fkey" FOREIGN KEY ("feedback_request_id") REFERENCES "Feedback_Request_Model"("id") ON DELETE SET NULL ON UPDATE CASCADE;
