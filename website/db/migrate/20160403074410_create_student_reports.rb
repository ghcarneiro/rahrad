class CreateStudentReports < ActiveRecord::Migration
  def change
    create_table :student_reports do |t|

      t.timestamps null: false
    end
  end
end
