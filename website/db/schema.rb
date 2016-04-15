# encoding: UTF-8
# This file is auto-generated from the current state of the database. Instead
# of editing this file, please use the migrations feature of Active Record to
# incrementally modify your database, and then regenerate this schema definition.
#
# Note that this schema.rb definition is the authoritative source for your
# database schema. If you need to create the application database on another
# system, you should be using db:schema:load, not running all the migrations
# from scratch. The latter is a flawed and unsustainable approach (the more migrations
# you'll amass, the slower it'll run and the greater likelihood for issues).
#
# It's strongly recommended that you check this file into your version control system.

ActiveRecord::Schema.define(version: 20160412225336) do

  create_table "dx_level1s", force: :cascade do |t|
    t.string   "name"
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
  end

  create_table "dx_level2s", force: :cascade do |t|
    t.string  "name"
    t.integer "dx_level1_id"
  end

  add_index "dx_level2s", ["dx_level1_id"], name: "index_dx_level2s_on_dx_level1_id"

  create_table "dx_level3s", force: :cascade do |t|
    t.string   "name"
    t.integer  "dx_level2_id"
    t.datetime "created_at",   null: false
    t.datetime "updated_at",   null: false
  end

  add_index "dx_level3s", ["dx_level2_id"], name: "index_dx_level3s_on_dx_level2_id"

  create_table "end_dxes", force: :cascade do |t|
    t.string   "name"
    t.string   "category"
    t.integer  "frequency"
    t.integer  "dxable_id"
    t.string   "dxable_type"
    t.datetime "created_at",  null: false
    t.datetime "updated_at",  null: false
  end

  add_index "end_dxes", ["dxable_type", "dxable_id"], name: "index_end_dxes_on_dxable_type_and_dxable_id"

  create_table "expert_reports", force: :cascade do |t|
    t.string   "report_number"
    t.string   "report_text"
    t.string   "report_image"
    t.integer  "end_dx_id"
    t.integer  "learner_info_id"
    t.integer  "times_attempted"
    t.integer  "correct_diagnosis"
    t.decimal  "difficulty"
    t.datetime "created_at",        null: false
    t.datetime "updated_at",        null: false
  end

  add_index "expert_reports", ["end_dx_id"], name: "index_expert_reports_on_end_dx_id"

  create_table "learner_dxes", force: :cascade do |t|
    t.string   "name"
    t.integer  "end_dx_id"
    t.integer  "user_id"
    t.boolean  "review_list"
    t.integer  "cases_attempted"
    t.integer  "correct_dx"
    t.integer  "excellent_cases"
    t.datetime "created_at",      null: false
    t.datetime "updated_at",      null: false
  end

  add_index "learner_dxes", ["end_dx_id"], name: "index_learner_dxes_on_end_dx_id"
  add_index "learner_dxes", ["user_id"], name: "index_learner_dxes_on_user_id"

  create_table "learner_infos", force: :cascade do |t|
    t.integer  "user_id"
    t.integer  "expert_report_id"
    t.datetime "created_at",       null: false
    t.datetime "updated_at",       null: false
  end

  add_index "learner_infos", ["expert_report_id"], name: "index_learner_infos_on_expert_report_id"
  add_index "learner_infos", ["user_id"], name: "index_learner_infos_on_user_id"

  create_table "reoccurences", force: :cascade do |t|
    t.string   "query"
    t.integer  "interval"
    t.string   "unit"
    t.datetime "time"
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.integer  "user_id"
  end

  create_table "saved_searches", force: :cascade do |t|
    t.string   "query"
    t.datetime "time"
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.integer  "user_id"
  end

  create_table "student_reports", force: :cascade do |t|
    t.string "report_text"
    t.boolean "diagnosis_found"
    t.integer "correct_sentences"
    t.integer "missing_sentences"
    t.integer  "expert_report_id"
    t.integer  "learner_dx_id"
    t.integer  "user_id"
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
  end

  create_table "users", force: :cascade do |t|
    t.string   "email",                  default: "", null: false
    t.string   "encrypted_password",     default: "", null: false
    t.string   "firstname"
    t.string   "lastname"
    t.string   "year_of_training"
    t.string   "reset_password_token"
    t.datetime "reset_password_sent_at"
    t.datetime "remember_created_at"
    t.integer  "sign_in_count",          default: 0,  null: false
    t.datetime "current_sign_in_at"
    t.datetime "last_sign_in_at"
    t.string   "current_sign_in_ip"
    t.string   "last_sign_in_ip"
    t.datetime "created_at",                          null: false
    t.datetime "updated_at",                          null: false
  end

  add_index "users", ["email"], name: "index_users_on_email", unique: true
  add_index "users", ["reset_password_token"], name: "index_users_on_reset_password_token", unique: true

end
